"""
Utilities for instrumenting a torch model.

Trace will hook one layer at a time.
TraceDict will hook multiple layers at once.
subsequence slices intervals from Sequential modules.
get_module, replace_module, get_parameter resolve dotted names.
set_requires_grad recursively sets requires_grad in module parameters.
"""

import contextlib
import inspect
from collections import OrderedDict
from torch import nn
import torch
import copy

class Trace(contextlib.AbstractContextManager):
    """
    To retain the output of the named layer during the computation of
    the given network:

        with Trace(net, 'layer.name') as ret:
            _ = net(inp)
            representation = ret.output

    A layer module can be passed directly without a layer name, and
    its output will be retained.  By default, a direct reference to
    the output object is returned, but options can control this:

        clone=True  - retains a copy of the output, which can be
            useful if you want to see the output before it might
            be modified by the network in-place later.
        detach=True - retains a detached reference or copy.  (By
            default the value would be left attached to the graph.)
        retain_grad=True - request gradient to be retained on the
            output.  After backward(), ret.output.grad is populated.

        retain_input=True - also retains the input.
        retain_output=False - can disable retaining the output.
        edit_output=fn - calls the function to modify the output
            of the layer before passing it the rest of the model.
            fn can optionally accept (output, layer) arguments
            for the original output and the layer name.
        edit_input=fn - similarly with edit_output for input.
            accept (input, layer) arguments.
        with_kwargs=True - edit_input and edit_output should receive key arguments.
        layer_func_replace=fn - replace the function of the layer.
            accept (*args, **kargs) arguments, where kargs has 'layer_name'.
        stop=True - throws a StopForward exception after the layer
            is run, which allows running just a portion of a model.
    """

    def __init__(
        self,
        module,
        layer=None,
        retain_input=False,
        retain_output=False,
        retain_input_before_edit=False,
        retain_output_before_edit=False,
        with_kwargs = True,
        clone=True,
        detach=True,
        retain_grad=False,
        edit_input=None,
        edit_output=None,
        layer_func_replace=None,
        stop=False,
    ):
        """
        Method to replace a forward method with a closure that
        intercepts the call, and tracks the hook so that it can be reverted.
        """
        retainer = self
        self.layer = layer 
        if layer is not None:
            module = get_module(module, layer)
        self.module = module
        if layer_func_replace is not None:
            def layer_func_replace_wrap(layer):
                def wrapped_layer_func_replace(*args, **kargs):
                    kargs['layer_name'] = layer
                    return layer_func_replace(*args, **kargs)
                return wrapped_layer_func_replace
            self.module_original_forward = module.forward
            module.forward = layer_func_replace_wrap(layer)

        def __forward_hook_func__(m, inpt, output):
            if retain_input:
                retainer.input = recursive_copy(
                    inpt, clone=clone, detach=detach, retain_grad=False)  
                # retain_grad applies to output only.
            if retain_output_before_edit:
                retainer.output_before_edit = recursive_copy(output, clone=clone, 
                    detach=detach, retain_grad=retain_grad)
            if edit_output:
                output = invoke_with_optional_args(
                    edit_output, output=output, layer=self.layer
                )
            if retain_output: 
                retainer.output = recursive_copy(output, clone=clone, 
                    detach=detach, retain_grad=retain_grad)
                # When retain_grad is set, also insert a trivial
                # copy operation.  That allows in-place operations
                # to follow without error.
                if retain_grad:
                    output = recursive_copy(retainer.output, clone=True, detach=False)
            if stop:
                raise StopForward()
            return output
        
        def __forward_pre_hook_func__(m, inpt):
            if retain_input_before_edit:
                retainer.input_before_edit = recursive_copy(
                    inpt, clone=clone, detach=detach, retain_grad=False)
            if edit_input:
                inpt = invoke_with_optional_args(edit_input, input=inpt, layer=self.layer)
            return inpt

        def forward_hook_k(m, args, kargs, output):
            return __forward_hook_func__(m, (args, kargs), output)

        def forward_hook(m, inputs, output):
            return __forward_hook_func__(m, inputs, output)
        
        def forward_pre_hook_k(m, args, kargs):
            v = __forward_pre_hook_func__(m, (args, kargs))
            if v != None:
                args, kargs = v
            return args, kargs

        def forward_pre_hook(m, args):
            return __forward_pre_hook_func__(m, args)

        if with_kwargs:
            self.forward_hook_handle = module.register_forward_hook(forward_hook_k, with_kwargs = True)
            self.forward_pre_hook_handle = module.register_forward_pre_hook(forward_pre_hook_k, with_kwargs = True)
        else: 
            self.forward_hook_handle = module.register_forward_hook(forward_hook, with_kwargs = False)
            self.forward_pre_hook_handle = module.register_forward_pre_hook(forward_pre_hook, with_kwargs = False)
        self.stop = stop

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()
        if self.stop and issubclass(type, StopForward):
            return True

    def close(self):
        self.forward_hook_handle.remove()
        self.forward_pre_hook_handle.remove()
        if hasattr(self, 'module_original_forward'):
            self.module.forward = self.module_original_forward


class TraceDict(OrderedDict, contextlib.AbstractContextManager):
    """
    To retain the output of multiple named layers during the computation
    of the given network:

        with TraceDict(net, ['layer1.name1', 'layer2.name2']) as ret:
            _ = net(inp)
            representation = ret['layer1.name1'].output

    If edit_output is provided, it should be a function that takes
    two arguments: output, and the layer name; and then it returns the
    modified output.

    Other arguments are the same as Trace.  If stop is True, then the
    execution of the network will be stopped after the last layer
    listed (even if it would not have been the last to be executed).
    """

    def __init__(
        self,
        module,
        layers=None,
        retain_input=False,
        retain_output=False,
        retain_input_before_edit=False,
        retain_output_before_edit=False,
        with_kwargs = True,
        clone=True,
        detach=True,
        retain_grad=False,
        edit_input=None,
        edit_output=None,
        layer_func_replace=None,
        stop=False,
    ): 
        self.stop = stop

        def flag_last_unseen(it):
            try:
                it = iter(it)
                prev = next(it)
                seen = set([prev])
            except StopIteration:
                return
            for item in it:
                if item not in seen:
                    yield False, prev
                    seen.add(item)
                    prev = item
            yield True, prev

        for is_last, layer in flag_last_unseen(layers):
            self[layer] = Trace(
                module=module,
                layer=layer,
                retain_input=retain_input,
                retain_output=retain_output,
                retain_input_before_edit=retain_input_before_edit,
                retain_output_before_edit=retain_output_before_edit,
                with_kwargs = with_kwargs,
                clone=clone,
                detach=detach,
                retain_grad=retain_grad,
                edit_input=edit_input, 
                edit_output=edit_output,
                layer_func_replace=layer_func_replace,
                stop=stop and is_last,
            )

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()
        if self.stop and issubclass(type, StopForward):
            return True

    def close(self):
        for layer, trace in reversed(self.items()):
            trace.close()




class StopForward(Exception):
    """
    If the only output needed from running a network is the retained
    submodule then Trace(submodule, stop=True) will stop execution
    immediately after the retained submodule by raising the StopForward()
    exception.  When Trace is used as context manager, it catches that
    exception and can be used as follows:

    with Trace(net, layername, stop=True) as tr:
        net(inp) # Only runs the network up to layername
    print(tr.output)
    """

    pass


def recursive_copy(x, clone=None, detach=None, retain_grad=None):
    """
    Copies a reference to a tensor, or an object that contains tensors,
    optionally detaching and cloning the tensor(s).  If retain_grad is
    true, the original tensors are marked to have grads retained.
    """
    if not clone and not detach and not retain_grad:
        return x
    if isinstance(x, torch.Tensor):
        if retain_grad:
            if not x.requires_grad:
                x.requires_grad = True
            x.retain_grad()
        elif detach:
            x = x.detach()
        if clone:
            x = x.clone()
        return x
    # Only dicts, lists, and tuples (and subclasses) can be copied.
    if isinstance(x, dict):
        return type(x)({k: recursive_copy(v) for k, v in x.items()})
    elif isinstance(x, (list, tuple)):
        return type(x)([recursive_copy(v) for v in x])
    else:
        assert False, f"Unknown type {type(x)} cannot be broken into tensors."


def subsequence(
    sequential,
    first_layer=None,
    last_layer=None,
    after_layer=None,
    upto_layer=None,
    single_layer=None,
    share_weights=False,
):
    """
    Creates a subsequence of a pytorch Sequential model, copying over
    modules together with parameters for the subsequence.  Only
    modules from first_layer to last_layer (inclusive) are included,
    or modules between after_layer and upto_layer (exclusive).
    Handles descent into dotted layer names as long as all references
    are within nested Sequential models.

    If share_weights is True, then references the original modules
    and their parameters without copying them.  Otherwise, by default,
    makes a separate brand-new copy.
    """
    assert (single_layer is None) or (
        first_layer is last_layer is after_layer is upto_layer is None
    )
    if single_layer is not None:
        first_layer = single_layer
        last_layer = single_layer
    first, last, after, upto = [
        None if d is None else d.split(".")
        for d in [first_layer, last_layer, after_layer, upto_layer]
    ]
    return hierarchical_subsequence(
        sequential,
        first=first,
        last=last,
        after=after,
        upto=upto,
        share_weights=share_weights,
    )


def hierarchical_subsequence(
    sequential, first, last, after, upto, share_weights=False, depth=0
):
    """
    Recursive helper for subsequence() to support descent into dotted
    layer names.  In this helper, first, last, after, and upto are
    arrays of names resulting from splitting on dots.  Can only
    descend into nested Sequentials.
    """
    assert (last is None) or (upto is None)
    assert (first is None) or (after is None)
    if first is last is after is upto is None:
        return sequential if share_weights else copy.deepcopy(sequential)
    assert isinstance(sequential, torch.nn.Sequential), (
        ".".join((first or last or after or upto)[:depth] or "arg") + " not Sequential"
    )
    including_children = (first is None) and (after is None)
    included_children = OrderedDict()
    # A = current level short name of A.
    # AN = full name for recursive descent if not innermost.
    (F, FN), (L, LN), (A, AN), (U, UN) = [
        (d[depth], (None if len(d) == depth + 1 else d))
        if d is not None
        else (None, None)
        for d in [first, last, after, upto]
    ]
    for name, layer in sequential._modules.items():
        if name == F:
            first = None
            including_children = True
        if name == A and AN is not None:  # just like F if not a leaf.
            after = None
            including_children = True
        if name == U and UN is None:
            upto = None
            including_children = False
        if including_children:
            # AR = full name for recursive descent if name matches.
            FR, LR, AR, UR = [
                n if n is None or n[depth] == name else None for n in [FN, LN, AN, UN]
            ]
            chosen = hierarchical_subsequence(
                layer,
                first=FR,
                last=LR,
                after=AR,
                upto=UR,
                share_weights=share_weights,
                depth=depth + 1,
            )
            if chosen is not None:
                included_children[name] = chosen
        if name == L:
            last = None
            including_children = False
        if name == U and UN is not None:  # just like L if not a leaf.
            upto = None
            including_children = False
        if name == A and AN is None:
            after = None
            including_children = True
    for name in [first, last, after, upto]:
        if name is not None:
            raise ValueError("Layer %s not found" % ".".join(name))
    # Omit empty subsequences except at the outermost level,
    # where we should not return None.
    if not len(included_children) and depth > 0:
        return None
    result = torch.nn.Sequential(included_children)
    result.training = sequential.training
    return result


def set_requires_grad(requires_grad, *models):
    """
    Sets requires_grad true or false for all parameters within the
    models passed.
    """
    for model in models:
        if isinstance(model, torch.nn.Module):
            for param in model.parameters():
                param.requires_grad = requires_grad
        elif isinstance(model, (torch.nn.Parameter, torch.Tensor)):
            model.requires_grad = requires_grad
        else:
            assert False, "unknown type %r" % type(model)


def get_module(model:nn.Module, name:str)->nn.Module:
    """
    Finds the named module within the given model.
    """
    for n, m in model.named_modules():
        if n == name:
            return m
    raise LookupError(name)


def get_parameter(model:nn.Module, name:str)->torch.Tensor:
    """
    Finds the named parameter within the given model.
    """
    for n, p in model.named_parameters():
        if n == name:
            return p
    raise LookupError(name)


def replace_module(model, name, new_module):
    """
    Replaces the named module within the given model.
    """
    if "." in name:
        parent_name, attr_name = name.rsplit(".", 1)
        model = get_module(model, parent_name)
    # original_module = getattr(model, attr_name)
    setattr(model, attr_name, new_module)


def invoke_with_optional_args(fn, *args, **kwargs):
    """
    Invokes a function with only the arguments that it
    is written to accept, giving priority to arguments
    that match by-name, using the following rules.
    (1) arguments with matching names are passed by name.
    (2) remaining non-name-matched args are passed by order.
    (3) extra caller arguments that the function cannot
        accept are not passed.
    (4) extra required function arguments that the caller
        cannot provide cause a TypeError to be raised.
    Ordinary python calling conventions are helpful for
    supporting a function that might be revised to accept
    extra arguments in a newer version, without requiring the
    caller to pass those new arguments.  This function helps
    support function callers that might be revised to supply
    extra arguments, without requiring the callee to accept
    those new arguments.
    """
    argspec = inspect.getfullargspec(fn)
    pass_args = []
    used_kw = set()
    unmatched_pos = []
    used_pos = 0
    defaulted_pos = len(argspec.args) - (
        0 if not argspec.defaults else len(argspec.defaults)
    )
    # Pass positional args that match name first, then by position.
    for i, n in enumerate(argspec.args):
        if n in kwargs:
            pass_args.append(kwargs[n])
            used_kw.add(n)
        elif used_pos < len(args):
            pass_args.append(args[used_pos])
            used_pos += 1
        else:
            unmatched_pos.append(len(pass_args))
            pass_args.append(
                None if i < defaulted_pos else argspec.defaults[i - defaulted_pos]
            )
    # Fill unmatched positional args with unmatched keyword args in order.
    if len(unmatched_pos):
        for k, v in kwargs.items():
            if k in used_kw or k in argspec.kwonlyargs:
                continue
            pass_args[unmatched_pos[0]] = v
            used_kw.add(k)
            unmatched_pos = unmatched_pos[1:]
            if len(unmatched_pos) == 0:
                break
        else:
            if unmatched_pos[0] < defaulted_pos:
                unpassed = ", ".join(
                    argspec.args[u] for u in unmatched_pos if u < defaulted_pos
                )
                raise TypeError(f"{fn.__name__}() cannot be passed {unpassed}.")
    # Pass remaining kw args if they can be accepted.
    pass_kw = {
        k: v
        for k, v in kwargs.items()
        if k not in used_kw and (k in argspec.kwonlyargs or argspec.varargs is not None)
    }
    # Pass remaining positional args if they can be accepted.
    if argspec.varargs is not None:
        pass_args += list(args[used_pos:])
    return fn(*pass_args, **pass_kw)
