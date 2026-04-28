# HSTU Perfsim/Smart 使用说明

这个目录用于给 HSTU SM120 FP8 warp-specialized 前向 kernel 采集 CUDA APIC trace，并提交给 `flow.smart` 跑 Perfsim/Smart 分析。

当前推荐流程是两步法：

1. 在 GB202/SM120 compute node 的 docker/srun 环境里采集 `cuda.tgz`。
2. 在 login node 上把预采集的 `cuda.tgz` 交给 `flow.smart`。

不要在 compute node 的 docker 里直接跑 `flow.smart`。`flow.smart` 应该在 login node 上执行，例如 `computelab-frontend` 或 `sc-xterm`。

## 文件说明

- `collect_hstu_trace.sh`：在 GB202/SM120 机器上采集 HSTU CUDA APIC trace。
- `launch_smart_hstu.sh`：在 login node 上提交预采集 trace 给 `flow.smart`。
- `run_hstu_kernel_once.py`：构造 HSTU FP8 输入并运行目标 kernel。
- `run_hstu_kernel_once.sh`：给 workload yml 使用的 shell wrapper。
- `workload_hstu_sm120.yml`：workload 模式示例；当前不推荐作为主流程。
- `config.smart.yml`：当前使用的 Smart config，来自 `trtllm-gen/config.smart.yml`。
- `traces/`：本地 trace 输出目录，`cuda.tgz` 很大，通常不要提交到 git。
- `results/`：Smart/perfsim 输出目录，通常不要提交到 git。

`config.smart.tllmgen.yml` 不再是当前 HSTU perfsim 流程的默认 config。`launch_smart_hstu.sh` 默认使用 `perfsim/config.smart.yml`。

## 采集 CUDA APIC Trace

在 GB202/SM120 compute node 的 docker/srun 环境里执行：

```bash
cd /home/scratch.minyu_gpu/project/shopee/fbgemm-hstu
./perfsim/collect_hstu_trace.sh
```

默认参数：

- `bs=8`
- `seq=4096`
- `heads=16`
- `headdim=128`
- causal attention
- `frange=3:3`

默认捕获的 kernel 名称匹配：

```text
hstu_fwd_kernel_sm120_fp8_ws_tma
```

采集成功后会生成类似路径：

```bash
/home/minyu/project/shopee/fbgemm-hstu/perfsim/traces/hstu_sm120_fp8_bs8_seq4096_h16_causal_<commit>_<timestamp>/CUDA_APIC_TRACES/k0404_flash__hstu_fwd_kernel_sm120_fp8_ws_tma/cuda.tgz
```

脚本还会更新：

```bash
/home/minyu/project/shopee/fbgemm-hstu/perfsim/traces/hstu_latest_cuda.tgz
```

这个 symlink 指向最新采集到的 `cuda.tgz`。

常用变体：

```bash
./perfsim/collect_hstu_trace.sh --bs 4 --seq 2048
```

```bash
./perfsim/collect_hstu_trace.sh --full
```

```bash
./perfsim/collect_hstu_trace.sh --frange 2:2
```

## 提交 Smart

在 login node 上执行，不要在 compute node docker 里执行：

```bash
cd ~/project/shopee/fbgemm-hstu
./perfsim/launch_smart_hstu.sh --chip gb202 --trace /home/minyu/project/shopee/fbgemm-hstu/perfsim/traces/hstu_sm120_fp8_bs8_seq4096_h16_causal_<commit>_<timestamp>/CUDA_APIC_TRACES/k0404_flash__hstu_fwd_kernel_sm120_fp8_ws_tma/cuda.tgz
```

如果 `perfsim/traces/hstu_latest_cuda.tgz` 已经指向你要跑的 trace，也可以直接：

```bash
cd ~/project/shopee/fbgemm-hstu
./perfsim/launch_smart_hstu.sh --chip gb202
```

如果需要避开某些有硬件问题的机器，可以指定 node：

```bash
./perfsim/launch_smart_hstu.sh --chip gb202 --node <healthy-computelab-node>
```

脚本实际执行的是：

```bash
/home/scratch.svc_compute_arch/release/flow.smart/latest/flow.smart run \
  -chip gb202 \
  -pic \
  -cuda2ctl \
  -enableMorph \
  -trace <cuda.tgz> \
  -dir <output_dir> \
  -config /home/minyu/project/shopee/fbgemm-hstu/perfsim/config.smart.yml \
  -useSmart2
```

提交后可以在 Compute Nexus 查看：

```text
https://compute-nexus.nvidia.com/workflows/runs
```

完成后通常会收到邮件，结果目录会写到 `perfsim/results/`。

## 为什么使用两步法

不要把 HSTU binary 直接通过 workload 交给 `flow.smart` 运行，原因是 workload 模式会让 `flow.smart` 在它调度出来的节点上启动 workload。实际使用中容易遇到以下问题：

- login node 和 compute node/docker 的 glibc 或 CUDA runtime 环境不一致。
- Python/PyTorch/HSTU extension 路径在调度节点上不一致。
- `cudaReplayer` 与采集 trace 的 CUDA APIC 版本不一致。

两步法把“运行真实 CUDA workload”和“提交 Smart 模拟”分开：

- 真实 CUDA workload 在已知可用的 GB202/SM120 docker/srun 环境里运行。
- `flow.smart` 只消费已经生成好的 `cuda.tgz`。

这样可以绕开大部分 glibc 和 Python 环境差异。

## CUDA APIC 版本

`collect_hstu_trace.sh` 默认使用固定版本：

```bash
/home/scratch.svc_compute_arch/release/cuda_apic/linux64/release/0.1.2026041608261776353168/cuda_apic_capture.pl
```

不要随意换成 `latest`。之前遇到过采集端和 `flow.smart` 内部 `cudaReplayer` 版本不一致，导致 replay 或 gen_unify 阶段失败。

如果确实需要替换版本，用环境变量显式指定：

```bash
CUDA_APIC=/path/to/cuda_apic_capture.pl ./perfsim/collect_hstu_trace.sh
```

## PyTorch VMM 设置

采集脚本会设置：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
```

这是为了关闭 PyTorch expandable segments/VMM allocator，尽量使用 legacy `cudaMalloc` 路径，降低 `cudaReplayer` 在 replay trace 时出错的概率。

## Workload 模式

`workload_hstu_sm120.yml` 只是保留为参考，不是当前推荐流程。

如果要试 workload 模式，命令形态类似：

```bash
/home/scratch.svc_compute_arch/release/flow.smart/latest/flow.smart run \
  -chip gb202 \
  -pic \
  -cuda2ctlAmodel \
  -enableMorph \
  -workload /home/minyu/project/shopee/fbgemm-hstu/perfsim/workload_hstu_sm120.yml \
  -dir <output_dir> \
  -config /home/minyu/project/shopee/fbgemm-hstu/perfsim/config.smart.yml \
  -useSmart2
```

但如果 workload 需要依赖当前 docker 内的 Python/PyTorch/HSTU extension，优先使用预采集 trace 的两步法。

## Git 注意事项

`perfsim/traces/` 和 `perfsim/results/` 可能包含很大的 trace 和结果文件，通常不要提交。

当前需要提交的脚本/config 通常是：

```bash
git add perfsim/collect_hstu_trace.sh \
        perfsim/config.smart.yml \
        perfsim/launch_smart_hstu.sh \
        perfsim/run_hstu_kernel_once.py \
        perfsim/run_hstu_kernel_once.sh \
        perfsim/workload_hstu_sm120.yml \
        perfsim/README.md
```
