export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=1
python ensemble_attack.py --origin_sample_path "/data/hdd3/duhao/data/datasets/attack_dataset/clean_cls_samples" --attack_sample_path "./adv_samples"