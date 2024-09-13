# 构建镜像
docker build -t attack-alg:0.3 .

    -e HF_ENDPOINT=https://hf-mirror.com  \
    --shm-size=8g \
# 验证方式：运行容器时自动执行命令
docker run --rm \
    -v /data/hdd3/duhao/data/datasets/attack_dataset/clean_cls_samples:/data/hdd3/duhao/data/datasets/attack_dataset/clean_cls_samples \
    -v /data/hdd3/duhao/code/adversarial_competition/adv_samples:/data/hdd3/duhao/code/adversarial_competition/adv_samples \
    -e CUDA_VISIBLE_DEVICES=1  \
    --gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_VISIBLE_DEVICES=all \
    attack-alg:0.3 \
    --origin_sample_path=/data/hdd3/duhao/data/datasets/attack_dataset/clean_cls_samples \
    --attack_sample_path=/data/hdd3/duhao/code/adversarial_competition/adv_samples \
    --sample_type=1



