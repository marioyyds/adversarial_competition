export HF_ENDPOINT=https://hf-mirror.com
python train.py --gpu "1,2,3" --arch "swint" --batch_size 512
# python train.py --gpu "1,2,3" --arch "mobilenet_v2" --batch_size 256
# python train.py --gpu "1,2,3" --arch "mobilenet_v3" --batch_size 1024
# python train.py --gpu "0" --arch "vgg16" --batch_size 128