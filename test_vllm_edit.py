#%%
from utils import get_full_model_name, load_vllm_editor
from evaluation.vllm_editor_eval import VLLMEditorEvaluation
from utils.GLOBAL import ROOT_PATH
import os, argparse, sys

def get_attr():
    parser = argparse.ArgumentParser()
    parser.add_argument('-en', '--editor_name', type=str, help='Editor name: LiveEdit, FT_VL...', required=True)
    parser.add_argument('-mn', '--edit_model_name', type=str, help='Editing model name: llava...', required=True)
    parser.add_argument('-sen', '--sequential_edit_n', type=int, help='Edit number.', required=True)
    parser.add_argument('-enp', '--eval_name_postfix', type=str, default = '', help='Postfix name of this evaluation.')
    parser.add_argument('-dvc', '--device', type=str, help='CUDA device for editing.', required=True)
    parser.add_argument('-ckpt', '--editor_ckpt_path', type=str, default = None, help='For Editors that needs training.')
    parser.add_argument('-dn', '--data_name', type=str, required = True, help = 'Evaluating dataset, including EVQA, EIC.')
    parser.add_argument('-dsn', '--data_sample_n', type=int, default = None, help = 'Sample number for evaluation.')
    args = parser.parse_args()
    return args
 
class cfg:
    editor_name = 'liveedit'
    edit_model_name = 'blip2'
    sequential_edit_n = 100
    eval_name_postfix = 'test'
    device = 0
    editor_ckpt_path = 'records/...'
    data_name = 'EVQA' # 'EVQA', 'EIC'
    data_sample_n = 100 # 231, 530, 1300, 3000

if __name__ == '__main__':
    cfg = get_attr()
    cfg.editor_name = cfg.editor_name.lower()
    cfg.edit_model_name = get_full_model_name(cfg.edit_model_name)
    cfg.evaluation_name = cfg.data_name.upper()
    if cfg.eval_name_postfix != '':
        cfg.evaluation_name = '%s-%s'%(cfg.evaluation_name, cfg.eval_name_postfix)
    # if has evaluated, skip
    eval_result_dir_path = os.path.join('eval_results', cfg.editor_name, cfg.edit_model_name, cfg.evaluation_name, 'single_edit')
    if os.path.exists(eval_result_dir_path):
        print('Has evaluated: %s'%eval_result_dir_path)
        sys.exit()
    print(cfg)
    editor = load_vllm_editor(cfg.editor_name, cfg.edit_model_name, cfg.device, None, cfg.editor_ckpt_path, False)
    # load data
    if cfg.data_name == 'EVQA':
        from dataset.vllm import EVQA
        data_path = os.path.join(ROOT_PATH, 'data/easy-edit-mm/vqa/vqa_eval.json')
        img_root_dir = os.path.join(ROOT_PATH, 'data/easy-edit-mm/images')
        eval_data = EVQA(data_path, img_root_dir, cfg.data_sample_n)
    elif cfg.data_name == 'EIC':
        from dataset.vllm import EIC
        data_path = os.path.join(ROOT_PATH, 'data/easy-edit-mm/caption/caption_eval_edit.json')
        img_root_dir = os.path.join(ROOT_PATH, 'data/easy-edit-mm/images')
        eval_data = EIC(data_path, img_root_dir, cfg.data_sample_n)
    elif cfg.data_name == 'VLKEB':
        from dataset.vllm import VLKEB
        data_path = os.path.join(ROOT_PATH, 'data/VLKEB/eval.json')
        img_root_dir = os.path.join(ROOT_PATH, 'data/VLKEB/mmkb_images')
        eval_data = VLKEB(data_path, img_root_dir, cfg.data_sample_n)
    # evaluate
    ev = VLLMEditorEvaluation(editor, eval_data, cfg.evaluation_name, 'eval_results')
    ev.evaluate_sequential_edit(cfg.sequential_edit_n, False, None)

