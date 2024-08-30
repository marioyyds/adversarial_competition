export CUDA_VISIBLE_DEVICES=0
python metric.py --adv_sample '/data/hdd3/duhao/data/datasets/attack_dataset/adv_samples-64_8'\
                 --clean_sample '/data/hdd3/duhao/data/datasets/attack_dataset/clean_cls_samples'