export HF_ENDPOINT=https://hf-mirror.com
# python train.py --gpu "1,2,3" --arch "swint" --batch_size 512 --lr 0.001 --epoch 10
python train.py --gpu "1,2,3" --arch "efficientnet_b0.ra4_e3600_r224_in1k" --batch_size 512 --lr 0.001 --epoch 20
# python train.py --gpu "1,2,3" --arch "mobilenet_v2" --batch_size 256 --lr 0.1
# python train.py --gpu "1,2,3" --arch "mobilenet_v3" --batch_size 1024 --lr 0.1
# python train.py --gpu "0" --arch "vgg16" --batch_size 128 --lr 0.1