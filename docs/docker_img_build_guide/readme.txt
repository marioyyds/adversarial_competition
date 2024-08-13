# 构建镜像
docker build -t attack-alg:0.1 .

# 验证方式：运行容器时自动执行命令
docker run --rm attack-alg:0.1 --origin_sample_path=/data/dataset/attack/clean_cls_samples --attack_sample_path=/data/result/attack/20240627/adv_samples --sample_type=1


