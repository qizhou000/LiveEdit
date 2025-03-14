from torch import nn
import torch

class IDMLP(nn.Module):
    def __init__(
        self,
        indim: int,
        outdim: int,
        hidden_dim: int,
        n_hidden: int,
        init: str = None,
        act: str = None,
        rank: int = None,
        n_modes: int = None
    ):
        super().__init__()
        print(f"Building IDMLP ({init}) {[indim] * (n_hidden + 2)}")
        self.layers = nn.ModuleList(
            [
                LRLinear(indim, indim, rank=rank, relu=idx < n_hidden, init=init, n_modes=n_modes)
                for idx in range(n_hidden + 1)
            ]
        )

    def forward(self, x, mode=None):
        for layer in self.layers:
            x = layer(x, mode=mode)
        return x


class LRLinear(nn.Module):
    def __init__(self, inf, outf, rank: int = None, relu=False, init="id", n_modes=None):
        super().__init__()

        mid_dim = min(rank, inf)
        if init == "id":
            self.u = nn.Parameter(torch.zeros(outf, mid_dim))
            self.v = nn.Parameter(torch.randn(mid_dim, inf))
        elif init == "xavier":
            self.u = nn.Parameter(torch.empty(outf, mid_dim))
            self.v = nn.Parameter(torch.empty(mid_dim, inf))
            nn.init.xavier_uniform_(self.u.data, gain=nn.init.calculate_gain("relu"))
            nn.init.xavier_uniform_(self.v.data, gain=1.0)
        else:
            raise ValueError(f"Unrecognized initialization {init}")

        if n_modes is not None:
            self.mode_shift = nn.Embedding(n_modes, outf)
            self.mode_shift.weight.data.zero_()
            self.mode_scale = nn.Embedding(n_modes, outf)
            self.mode_scale.weight.data.fill_(1)

        self.n_modes = n_modes
        self.bias = nn.Parameter(torch.zeros(outf))
        self.inf = inf
        self.init = init

    def forward(self, x, mode=None):
        if mode is not None:
            assert self.n_modes is not None, "Linear got a mode but wasn't initialized for it"
            assert mode < self.n_modes, f"Input mode {mode} outside of range {self.n_modes}"
        assert x.shape[-1] == self.inf, f"Input wrong dim ({x.shape}, {self.inf})"

        pre_act = (self.u @ (self.v @ x.T)).T
        if self.bias is not None:
            pre_act += self.bias

        if mode is not None:
            if not isinstance(mode, torch.Tensor):
                mode = torch.tensor(mode).to(x.device)
            scale, shift = self.mode_scale(mode), self.mode_shift(mode)
            pre_act = pre_act * scale + shift

        # need clamp instead of relu so gradient at 0 isn't 0
        acts = pre_act.clamp(min=0)
        if self.init == "id":
            return acts + x
        else:
            return acts

 


def update_counter(x, m, s, k):
    new_m = m + (x - m) / k
    new_s = s + (x - m) * (x - new_m)
    return new_m, new_s


class GradientTransform(nn.Module):
    def __init__(self, x_dim: int, delta_dim: int, cfg, n_modes = None):
        super().__init__()

        self.x_dim = x_dim
        self.delta_dim = delta_dim
        self.cfg = cfg

        self.norm_init = False
        self.register_buffer("u_mean", torch.full((x_dim,), float("nan")))
        self.register_buffer("v_mean", torch.full((delta_dim,), float("nan")))
        self.register_buffer("u_std", torch.full((x_dim,), float("nan")))
        self.register_buffer("v_std", torch.full((delta_dim,), float("nan")))
        self.register_buffer("u_s", torch.full((x_dim,), float("nan")))
        self.register_buffer("v_s", torch.full((delta_dim,), float("nan")))
        self.register_buffer("k", torch.full((1,), float("nan")))

        self.mlp = IDMLP(delta_dim + x_dim, delta_dim + x_dim, (delta_dim + x_dim) * 2,
                            cfg.n_hidden, init=cfg.init, act=cfg.act, rank=cfg.rank, n_modes=n_modes)

    def forward(self, u, v, param_idx=None):
        u, v = u.to(torch.float32), v.to(torch.float32)

        u_ = u.view(-1, u.shape[-1]) # [rep_num, dim_in]
        v_ = v.view(-1, v.shape[-1]) # [rep_num, dim_out]
        nz_mask = (u_ != 0).any(-1) * (v_ != 0).any(-1)  # Skip batch elements with zero grad
        u_ = u_[nz_mask]
        v_ = v_[nz_mask]
        # print(v_.shape)
        # print(v_.shape)
        # raise

        if self.training:
            for idx in range(u_.shape[0]):
                if not self.norm_init:
                    self.u_mean = u_[idx].clone().detach()
                    self.v_mean = v_[idx].clone().detach()
                    self.u_s.zero_()
                    self.v_s.zero_()
                    self.k[:] = 1
                    self.norm_init = True
                else:
                    self.k += 1
                    self.u_mean, self.u_s = update_counter(u_[idx], self.u_mean, self.u_s, self.k)
                    self.v_mean, self.v_s = update_counter(v_[idx], self.v_mean, self.v_s, self.k)

            if self.k < 2:
                raise RuntimeError(f"Can't perform normalization with only {self.k} samples so far")
            self.u_std = (self.u_s / (self.k - 1)) ** 0.5
            self.v_std = (self.v_s / (self.k - 1)) ** 0.5

        if self.cfg.norm:
            u_input = (u_ - self.u_mean) / (self.u_std + 1e-7)
            v_input = (v_ - self.v_mean) / (self.v_std + 1e-7)
        else:
            u_input = u_
            v_input = v_

        # if self.cfg.combine:
        output = self.mlp(torch.cat((u_input, v_input), -1), mode=param_idx)
        out1, out2 = output.split([u.shape[-1], v.shape[-1]], -1)
        return out1, out2
