export HF_ENDPOINT=https://hf-mirror.com
python patch_attack.py --netClassifier "resnet18_patch" --train_set '/data/hdd3/duhao/data/datasets/attack_dataset/clean_cls_samples' --test_set '/data/hdd3/duhao/data/datasets/attack_dataset/clean_cls_samples'
