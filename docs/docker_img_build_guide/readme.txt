# 构建镜像
docker build -t attack-alg:0.3 .

    -e HF_ENDPOINT=https://hf-mirror.com  \
    --shm-size=8g \
# 验证方式：运行容器时自动执行命令
docker run --rm \
    -v /data/hdd3/duhao/data/datasets/attack_dataset/clean_cls_samples:/data/hdd3/duhao/data/datasets/attack_dataset/clean_cls_samples \
    -v /data/hdd3/duhao/code/adversarial_competition/adv_samples:/data/hdd3/duhao/code/adversarial_competition/adv_samples \
    --gpus="device=0" \
    attack-alg:0.3 \
    --origin_sample_path=/data/hdd3/duhao/data/datasets/attack_dataset/clean_cls_samples \
    --attack_sample_path=/data/hdd3/duhao/code/adversarial_competition/adv_samples \
    --sample_type=1



