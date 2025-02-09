export HF_ENDPOINT=https://hf-mirror.com
# python eval.py --arch 'resnet34' --test_set '/home/heshiyuan/code/adversarial_competition/data/adv_samples' --batch_size 32 --gpu "2"
# python eval.py --arch 'resnet50.a1_in1k' --test_set '/data2/huhongx/adversarial_competition/attack_dataset/adv_samples_3' --batch_size 32 --gpu "2"
# python eval.py --arch 'resnet101.a1h_in1k' --test_set '/home/heshiyuan/code/adversarial_competition/data/adv_samples' --batch_size 32 --gpu "2"
python eval.py --arch 'ens_adv_inception_resnet_v2' --test_set '/data2/huhongx/adversarial_competition/attack_dataset/clean_cls_samples' --batch_size 1 --gpu "2" --num_worker 0
# python eval.py --arch 'vit' --test_set '/data2/huhongx/adversarial_competition/attack_dataset/phase1/train_set'
# python eval.py --arch 'efficientnet_b0.ra4_e3600_r224_in1k' --test_set '/data2/huhongx/adversarial_competition/attack_dataset/phase1/test_set'