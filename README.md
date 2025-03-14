# Setup
1. Please download the MMEdit datasets from the URL provided in [1] and place the related folders in the `data/easy-edit-mm` directory.
2. Please download the VLKEB dataset from the URL provided in [2] and place the related folders in the `data/VLKEB` directory.
3. Please modify the `ROOT_PATH` in `utils/GLOBAL.py` to the absolute path of the current directory, and update `model_path_map` to the absolute paths of each backbone's weights.

# Train editors
Please follow the script below to train an editor that requires training:

`python train_vllm_editor.py -en liveedit -mn blip2 -dna EVQA -bs 8 -dvc "cuda:0" -edvc 0 -lkpt None -tnp EVQA -eps 50 -sci 500`

# Evaluate editors
Please follow the script below to evaluate the editors. If an editor does not require training, please set `-ckpt` to `None`."

`python test_vllm_edit.py -en "liveedit" -mn "blip2" -sen 1000 -dvc "cuda:0" -dn "EVQA" -ckpt "records/liveedit/blip2-opt-2.7b/EVQA/checkpoints/ckpt"`

[1] Can We Edit Multimodal Large Language Models? EMNLP 2023

[2] VLKEB: A Large Vision-Language Model Knowledge Editing Benchmark. NeurIPS 2024, Datasets and Benchmarks Track
