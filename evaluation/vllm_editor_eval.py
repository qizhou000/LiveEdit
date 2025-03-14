from editor.vllm_editors.base import VLLMBaseEditor
from editor.vllms_for_edit.base import BaseVLLMForEdit
from dataset.vllm import BaseVLLMEditData
from typing import List, Dict, Union
from collections import defaultdict
from datetime import datetime
from copy import deepcopy
import torch, os, json
from tqdm import tqdm
from time import time
import numpy as np

class VLLMEditorEvaluation():
    def __init__(self, editor:VLLMBaseEditor, eval_data:BaseVLLMEditData, 
        evaluation_name = None, results_dir = 'eval_results') -> None:
        '''
        `results_dir` & `evaluation_name`: Used to create result directory.
            `evaluation_name` can be set as dataset name.
        '''
        self.editor = editor
        self.eval_data = eval_data
        editor_name, model_name = editor.name_of_editor_and_model()
        t = datetime.now().strftime('%Y.%m.%d-%H.%M.%S')
        evaluation_name = evaluation_name if evaluation_name else t
        self.result_dir = os.path.join(results_dir, editor_name, model_name, evaluation_name)
        print('Evaluation results directory: ', self.result_dir)
        # self.eval_xyms = None

    def evaluate_single_edit(self):
        editor = self.editor
        print('Evaluating reliability, generality and locality for %s on %s with single editing.'
              %editor.name_of_editor_and_model())
        eval_data = deepcopy(self.eval_data.data_with_img)
        for ed in eval_data: # single edit, the number of requests must be 1.
            assert len(ed['requests']) == 1
        result_data = deepcopy(self.eval_data.data_with_img_path)
        tokenizer = editor.vllm.get_llm_tokenizer()
        editor.restore_to_original_model()  
        results = [] 
        for rd, ed in zip(tqdm(result_data, 'Evaluating'), eval_data):
            rd['reliability'] = rd.pop('requests') 
            rd['reliability'][0]['target'] = rd['reliability'][0].pop('target_new')
            # predict before edit for locality data
            for loc_name in ed['locality'].keys():
                for rdl, edl in zip(rd['locality'][loc_name], ed['locality'][loc_name]):
                    (input_embeds, vt_range), label_ids, label_masks = editor.vllm.prompts_imgs_target_to_xym(
                        [edl['prompt']], [edl['image']], [edl['target']])
                    logits = editor.vllm.get_llm_outpt(input_embeds, vt_range).logits
                    before_edit_ids = torch.softmax(logits, -1).argmax(-1)[:, -label_ids.shape[1]:] # [1, l2]
                    rdl['predict_before_edit'] = tokenizer.decode(label_ids[label_masks.to(bool)])
                    edl['before_edit_ids'] = before_edit_ids
            # edit 
            start_t = time()
            editor.edit_one_piece(ed['requests'][0])
            rd['reliability'][0]['edit_time'] = time() - start_t
            # compute scores 
            rd = self.__get_results_after_edit__(editor.vllm, ed, rd)
            results.append(rd)
            # Restore to original model
            editor.restore_to_original_model()
        save_dir = os.path.join(self.result_dir, 'single_edit')
        # save results
        self.save_results(os.path.join(save_dir, 'results.json'), results)
        mean_results = self.get_mean_results(results)
        mean_results['sample_count'] = len(results)
        self.save_results(os.path.join(save_dir, 'mean_results.json'), mean_results)
        return results

    def evaluate_sequential_edit(self, edit_n = 10, random = False, seed = None):
        editor = self.editor
        print('Evaluating reliability, generality and locality for %s on %s with sequential editing %s.'
              %(*editor.name_of_editor_and_model(), edit_n))
        # preprocess data for sequential editing evaluation
        def split_data(data): 
            splited_data = []
            splited_data_ns = []
            now_split = []
            now_split_edit_n = 0
            for d in data:
                now_split.append(d)
                now_split_edit_n += len(d['requests'])
                if now_split_edit_n >= edit_n:
                    splited_data.append(now_split)
                    splited_data_ns.append(now_split_edit_n)
                    now_split = []
                    now_split_edit_n = 0
            return splited_data, splited_data_ns
        eval_data = deepcopy(self.eval_data.data_with_img)
        result_data = deepcopy(self.eval_data.data_with_img_path)
        if random:
            seed = seed if seed != None else np.random.randint(1, 999999)
            np.random.default_rng(seed).shuffle(eval_data)
            np.random.default_rng(seed).shuffle(result_data)
        eval_data, eval_data_ns = split_data(eval_data)
        result_data, _ = split_data(result_data)
        # evaluate
        tokenizer = editor.vllm.get_llm_tokenizer()
        editor.restore_to_original_model()
        results = [] 
        for split_rd, split_ed in zip(tqdm(result_data, 'Evaluating'), eval_data):
            split_res = []
            for rd, ed in zip(tqdm(split_rd, 'Preparing', leave = False), split_ed):
                rd['reliability'] = rd.pop('requests') 
                for r in rd['reliability']:
                    r['target'] = r.pop('target_new') 
                for loc_name in ed['locality'].keys(): # predict before edit for locality data
                    for rdl, edl in zip(rd['locality'][loc_name], ed['locality'][loc_name]):
                        (input_embeds, vt_range), label_ids, label_masks = editor.vllm.prompts_imgs_target_to_xym(
                            [edl['prompt']], [edl['image']], [edl['target']])
                        logits = editor.vllm.get_llm_outpt(input_embeds, vt_range).logits
                        before_edit_ids = torch.softmax(logits, -1).argmax(-1)[:, -label_ids.shape[1]:] # [1, l2]
                        rdl['predict_before_edit'] = tokenizer.decode(before_edit_ids[label_masks.to(bool)])
                        edl['before_edit_ids'] = before_edit_ids
            for rd, ed in zip(tqdm(split_rd, 'Editing', leave = False), split_ed): # edit 
                for rdr, edr in zip(rd['reliability'], ed['requests']):
                    start_t = time()
                    editor.edit_one_piece(edr)
                    rdr['edit_time'] = time() - start_t
            for rd, ed in zip(tqdm(split_rd, 'Testing', leave = False), split_ed): # compute scores 
                rd = self.__get_results_after_edit__(editor.vllm, ed, rd)
                split_res.append(rd)
            editor.restore_to_original_model()
            results.append(split_res)
        # save results
        save_dir = os.path.join(self.result_dir, 'sequential_edit_%s'%edit_n)
        self.save_results(os.path.join(save_dir, '%sresults.json'%('seed_%s_'%seed if random else '')), results)
        split_mean = [self.get_mean_results(sr) for sr in results]
        for mr, n in zip(split_mean, eval_data_ns):
            mr['sequential_edit_n'] = n
        total_mean = self.get_mean_results([r for sr in results for r in sr])
        total_mean['total_edit_n'] = sum(eval_data_ns)
        mean_results = {"total_mean": total_mean, "split_mean": split_mean}
        self.save_results(os.path.join(save_dir, '%smean_results.json'%('seed_%s_'%seed if random else '')), mean_results)
        return results

    def __get_results_after_edit__(self, vllm:BaseVLLMForEdit, ed, rd):
        def get_eval_xym(prompt, image, target):
            (x, vt_range), y, m = vllm.prompts_imgs_target_to_xym([prompt], [image], [target])
            x['query_triple'] = (prompt, image, target)
            x['query_range'] = (0, x['inputs_embeds'].shape[1] - m.shape[1] + 1)
            return (x, vt_range), y, m
        def accuracy_and_prediction(input_embeds, vt_range, label_ids, label_masks):
            # label_ids/label_masks: [1, l2]
            assert len(label_ids) == 1 and len(label_masks) == 1
            logits = vllm.get_llm_outpt(input_embeds, vt_range).logits # [1,l1,d]
            pre_y = torch.softmax(logits, -1).argmax(-1) # [1, l1]
            pre_y = pre_y[:, -label_ids.shape[1]:] # [1, l2]
            acc = ((pre_y == label_ids) * label_masks).sum()/label_masks.sum() 
            return float(acc), pre_y
        tokenizer = vllm.get_llm_tokenizer()
        # reliability
        for rdr, edr in zip(rd['reliability'], ed['requests']):
            (input_embeds, vt_range), label_ids, label_masks = get_eval_xym(
                    edr['prompt'], edr['image'], edr['target_new'])
            acc, pre_y = accuracy_and_prediction(input_embeds, vt_range, label_ids, label_masks)
            rdr['predict_after_edit'] = tokenizer.decode(pre_y[label_masks.to(bool)])
            rdr['acc'] = acc
        # generality
        for gen_name in ed['generality']:
            for rdg, edg in zip(rd['generality'][gen_name], ed['generality'][gen_name]):
                (input_embeds, vt_range), label_ids, label_masks = get_eval_xym(
                    edg['prompt'], edg['image'], edg['target'])
                acc, pre_y = accuracy_and_prediction(input_embeds, vt_range, label_ids, label_masks)
                rdg['predict_after_edit'] = tokenizer.decode(pre_y[label_masks.to(bool)])
                rdg['acc'] = acc
        # locality
        for loc_name in ed['locality']:
            for rdl, edl in zip(rd['locality'][loc_name], ed['locality'][loc_name]):
                (input_embeds, vt_range), _, label_masks = get_eval_xym(
                    edl['prompt'], edl['image'], edl['target'])
                acc, pre_y = accuracy_and_prediction(input_embeds, vt_range, edl['before_edit_ids'], label_masks)
                rdl['predict_after_edit'] = tokenizer.decode(pre_y[label_masks.to(bool)])
                rdl['acc'] = acc
        return rd

    def get_mean_results(self, results:List[Dict]):
        """Get numbers from a result: {
            "reliability": [
                {"acc": float, "edit_time": float}, 
                {"acc": float, "edit_time": float}, ...]
            "generality": {
                sub_metric_1: [{"acc": float}, {"acc": float}, ...], 
                sub_metric_2: [{"acc": float}, {"acc": float}, ...], ...}
            "locality": {
                sub_metric_1: [{"acc": float}, {"acc": float}, ...], 
                sub_metric_2: [{"acc": float}, {"acc": float}, ...], ...}
        }
        """
        mean_res = {"reliability": {}, "generality": {}, "locality": {}}
        # sum values
        for r in results:
            for rr in r['reliability']:
                for value_name, value in rr.items():
                    if isinstance(value, (int, float)):
                        if value_name not in mean_res['reliability']:
                            mean_res['reliability'][value_name] = [0, 0]
                        mean_res['reliability'][value_name][0] += value
                        mean_res['reliability'][value_name][1] += 1
            for sub_metric in r['generality'].keys():
                if sub_metric not in mean_res['generality']:
                    mean_res['generality'][sub_metric] = {}
                for sub_res in r['generality'][sub_metric]:
                    for value_name, value in sub_res.items():
                        if isinstance(value, (int, float)):
                            if value_name not in mean_res['generality'][sub_metric]:
                                mean_res['generality'][sub_metric][value_name] = [0, 0]
                            mean_res['generality'][sub_metric][value_name][0] += value
                            mean_res['generality'][sub_metric][value_name][1] += 1
            for sub_metric in r['locality'].keys():
                if sub_metric not in mean_res['locality']:
                    mean_res['locality'][sub_metric] = {}
                for sub_res in r['locality'][sub_metric]:
                    for value_name, value in sub_res.items():
                        if isinstance(value, (int, float)):
                            if value_name not in mean_res['locality'][sub_metric]:
                                mean_res['locality'][sub_metric][value_name] = [0, 0]
                            mean_res['locality'][sub_metric][value_name][0] += value
                            mean_res['locality'][sub_metric][value_name][1] += 1
        # compute mean results
        for value_name, value in mean_res['reliability'].items():
            mean_res['reliability'][value_name] = value[0] / value[1]
        for sub_metric in mean_res['generality'].keys():
            for value_name, value in mean_res['generality'][sub_metric].items():
                mean_res['generality'][sub_metric][value_name] = value[0] / value[1]
        for sub_metric in mean_res['locality'].keys():
            for value_name, value in mean_res['locality'][sub_metric].items():
                mean_res['locality'][sub_metric][value_name] = value[0] / value[1]
        return mean_res

    def save_results(self, save_path:str, results:Dict, decimal = 4):
        def set_decimal(r):
            if isinstance(r, list):
                for i in range(len(r)):
                    r[i] = set_decimal(r[i])
            elif isinstance(r, dict) or isinstance(r, defaultdict):
                for k in r.keys():
                    r[k] = set_decimal(r[k])
            elif isinstance(r, float):
                r = round(r, decimal)
            return r
        res = deepcopy(results)
        res = set_decimal(res)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(os.path.join(save_path), 'w') as f:
            json.dump(res, f, indent = 4)


