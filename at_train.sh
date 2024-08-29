export HF_ENDPOINT=https://hf-mirror.com
python at_train.py --arch resnet50.a1_in1k-at --resume --lr 0.1 --gpu 3 --epoch 40