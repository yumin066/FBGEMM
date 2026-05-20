#!/usr/bin/env bash
set -euo pipefail

NV_GPU=$(nvidia-smi --format=csv,noheader --query-gpu=uuid | tr -sd , - | tr '\n' ' ')
IMAGE="${HSTU_COMPUTELAB_IMAGE:-hstu:minyu}"

if [[ $# -eq 0 ]]; then
        cmd=(/bin/bash)
        tty_args=(-it)
else
        cmd=("$@")
        tty_args=()
fi

if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
        DOCKER_CONFIG="${HSTU_COMPUTELAB_DOCKER_CONFIG:-/home/scratch.minyu_gpu/project/.docker}" \
        docker build \
             --tag "${IMAGE}" \
             -f dockerfile .  \
             --build-arg uid=$(id -u) --build-arg username=$(id -un) \
             --build-arg gid=$(id -g) --build-arg groupname=$(id -gn)
else
        echo "Using existing docker image: ${IMAGE}"
fi

#docker run --ipc=host --shm-size=512m --gpus 2 -it mcore:minyu

docker run -p 9527:22 --rm \
        --cap-add SYS_ADMIN --security-opt seccomp=unconfined \
        --ipc=host \
        --gpus all \
        --shm-size=512m \
        "${tty_args[@]}" \
        --name hstu-computelab-minyu \
        -v /home/scratch.minyu_gpu/:/home/scratch.minyu_gpu/ \
        -v /home/scratch.trt_llm_data/:/home/scratch.trt_llm_data/ \
        -v /home/scratch.svc_compute_arch/:/home/scratch.svc_compute_arch/ \
        -v /home/scratch.junyiq_gpu_1/:/home/scratch.junyiq_gpu_1/ \
        -v /home/minyu/:/home/minyu/ \
        -v /home/tools_ai/:/home/tools_ai/ \
        "${IMAGE}" \
        "${cmd[@]}"
