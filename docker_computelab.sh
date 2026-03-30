NV_GPU=$(nvidia-smi --format=csv,noheader --query-gpu=uuid | tr -sd , - | tr '\n' ' ')

docker build \
     --tag hstu:minyu \
     -f Dockerfile .  \
     --build-arg uid=$(id -u) --build-arg username=$(id -un) \
     --build-arg gid=$(id -g) --build-arg groupname=$(id -gn)

#docker run --ipc=host --shm-size=512m --gpus 2 -it mcore:minyu

docker run -p 9527:22 --rm \
        --cap-add SYS_ADMIN --security-opt seccomp=unconfined \
        --ipc=host \
        --gpus all \
        --shm-size=512m \
        -it \
        --name trtllm_minyu \
        -v /home/scratch.minyu_gpu/:/home/scratch.minyu_gpu/ \
        -v /home/scratch.trt_llm_data/:/home/scratch.trt_llm_data/ \
        -v /home/minyu/:/home/minyu/ \
        -v /home/tools_ai/:/home/tools_ai/ \
        hstu:minyu
