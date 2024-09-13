# 安装CUDA和cuDNN
# FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
FROM ubuntu:20.04

# E: The repository 'https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64  InRelease' is not signed.
RUN rm -rf /etc/apt/sources.list.d/* \
&& apt-get update -o Acquire::AllowInsecureRepositories=true --allow-releaseinfo-change\
&& apt-get upgrade -y --allow-unauthenticated

ENV DEBIAN_FRONTEND noninteractive

# apt源添加Python
RUN apt-get --no-install-recommends install -yq software-properties-common --allow-unauthenticated\
&& add-apt-repository ppa:deadsnakes/ppa -y\
&& sed -i "s@ppa.launchpadcontent.net@launchpad.proxy.ustclug.org@g" /etc/apt/sources.list /etc/apt/sources.list.d/*.list \
&& apt-get update 

# 安装Python3.10
COPY ./resources/get-pip.py /get-pip.py
RUN apt-get --no-install-recommends install -yq python3.10 python3-pip python3.10-distutils \
&& ln -sf /usr/bin/python3.10 /usr/bin/python3 \
&& ln -sf /usr/bin/python3.10 /usr/bin/python \
&& pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ \
&& pip config set global.timeout 60 \
# apt安装的python3-pip，/usr/bin/pip命令有问题，执行pip install会报错
&& python get-pip.py \
&& ln -sf /usr/local/bin/pip3 /usr/bin/pip3 \
&& ln -sf /usr/local/bin/pip /usr/bin/pip \
&& pip install -U pip setuptools wheel

# Python依赖通过源码的方式安装
RUN apt-get update \
&& apt-get install git make gcc g++ python3.10-dev -yq \
&& pip install cython

# 时区与语言
RUN apt-get --no-install-recommends install -yq tzdata language-pack-zh-hans \
&& ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime \
&& echo "Asia/Shanghai" > /etc/timezone \
&& localedef -c -f UTF-8 -i zh_CN zh_CN.UTF-8 \
&& echo 'LANG="zh_CN.UTF-8"' > /etc/locale.conf
ENV LANG zh_CN.UTF-8

# 解决图像识别运行报错, ImportError: libGL.so.1: cannot open shared object file: No such file or directory
RUN apt-get install -y libgl1-mesa-glx


# 拷贝算法代码（选手需修改）
COPY /utils /adversarial_competition/utils
COPY /models /adversarial_competition/models
COPY /ensemble_attack.py /adversarial_competition/ensemble_attack.py
COPY /requirements.txt /adversarial_competition/requirements.txt
COPY /checkpoint /adversarial_competition/checkpoint
COPY /docs/算法技术方案.doc /doc/算法技术方案.doc

# 配置终端的工作目录（选手需修改）
WORKDIR /adversarial_competition

# 安装算法所需依赖，Tensorflow、PyTorch等要注意匹配CUDA版本（选手需修改）
RUN pip install -r requirements.txt -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
# 配置运行容器时执行命令：python main.py（选手需修改）
ENTRYPOINT ["python", "ensemble_attack.py"]
# ENTRYPOINT ["bash"]
