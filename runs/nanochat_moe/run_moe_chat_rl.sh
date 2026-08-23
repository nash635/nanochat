#!/bin/bash
#
# Phase 5: MoE 强化学习 (Chat RL — GRPO/REINFORCE 简化版)
#
# 硬件: 8× NVIDIA H20-3e (Hopper SM 9.0, 96GB 每卡)
# 模型: 加载 MoE SFT checkpoint (d24, step 466, val_bpb=0.3104)
#   n_exp=8, top_k=2, stride=2 → 总参数 2.97B, active 1.61B
# 算法: on-policy REINFORCE (token-level advantage r - mu, DAPO 风格)
# 数据: GSM8K train (rollout 采样 + 沙盒验证) + test (pass@k eval)
#       (通过 symlink 共享 cache/ 的 task_data, 不重复下载)
#
# 前置条件:
#   - Phase 4 已完成 (cache_moe/chatsft_checkpoints/d24/ 有 model_000466.pt)
#   - cache/task_data/openai--gsm8k 已有 GSM8K 数据 (dense WP4 跑过即存在)
#   - /opt/venv/bin/python 可用
#
# 用法:
#   bash runs/nanochat_moe/run_moe_chat_rl.sh                        # 完整 RL (默认带 aux loss)
#   bash runs/nanochat_moe/run_moe_chat_rl.sh --no-aux-loss          # 关闭 aux loss 对比
#   bash runs/nanochat_moe/run_moe_chat_rl.sh --num-epochs 2         # 透传任意 chat_rl 参数
#
# 输出 (与 dense 完全隔离):
#   - Checkpoint: cache_moe/chatrl_checkpoints/d24/
#   - dense 的 cache/ 目录不受影响
#
# 注意 (docs/nanochat_moe.md §2.4): RL 阶段 aux loss 默认保留 (use_aux_loss=True),
#   作为 router 的独立正则项接入 (不混入逐 token logp)。可用 --no-aux-loss 关闭对比。

set -euo pipefail

# 定位 nanochat 根目录（本脚本位于 runs/nanochat_moe/ 子目录）
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# ===== 环境配置 =====
# MoE 产物统一落到 cache_moe/，tokenizer/GSM8K 数据通过 symlink 共享 cache/ 的 (不重复下载)
export NANOCHAT_BASE_DIR="$ROOT_DIR/cache_moe"
export NANOCHAT_DATASET_URL="https://hf-mirror.com"
export OMP_NUM_THREADS=1
export PATH=/opt/venv/bin:$PATH

CACHE_DIR="$ROOT_DIR/cache"

# ===== 建立隔离目录 + symlink 共享 tokenizer/GSM8K 数据 =====
mkdir -p "$NANOCHAT_BASE_DIR"
[ -e "$NANOCHAT_BASE_DIR/tokenizer" ] || ln -s "$CACHE_DIR/tokenizer" "$NANOCHAT_BASE_DIR/tokenizer"
[ -e "$NANOCHAT_BASE_DIR/task_data" ] || ln -s "$CACHE_DIR/task_data" "$NANOCHAT_BASE_DIR/task_data"

# ===== 打印配置概览 =====
echo "============================================"
echo " Phase 5: MoE 强化学习 (Chat RL)"
echo "============================================"
echo " 时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo " 硬件: 8× NVIDIA H20-3e"
echo " 模型: 加载 MoE SFT d24 (step 466, val_bpb=0.3104)"
echo " 数据: GSM8K train (rollout) + test (pass@k eval)"
echo " Batch: examples-per-step=16, num-samples=16"
echo " aux loss: 默认保留 (--no-aux-loss 可关)"
echo " Checkpoint 目录: cache_moe/chatrl_checkpoints/d24/"
echo "============================================"
echo ""

# ===== 环境检查 =====
echo "[1/3] 检查 Python 环境..."
/opt/venv/bin/python --version
/opt/venv/bin/python -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPUs: {torch.cuda.device_count()}')" 2>&1 | grep -v FutureWarning

echo "[2/3] 检查 GPU 可用性..."
nvidia-smi --query-gpu=index,name --format=csv,noheader 2>&1 | head -8

echo "[3/3] 检查 MoE SFT checkpoint..."
ls -1 cache_moe/chatsft_checkpoints/d24/model_000466.pt 2>/dev/null && echo "  [OK] SFT checkpoint 就绪" || {
    echo "  [FAIL] 未找到 MoE SFT checkpoint, 请先完成 Phase 4"
    exit 1
}
ls -1 cache/task_data/openai--gsm8k/main/train/00000.parquet 2>/dev/null && echo "  [OK] GSM8K 数据就绪" || {
    echo "  [WARN] GSM8K 缺失, 首次运行会从 hf-mirror.com 下载"
}

echo ""
echo "============================================"
echo " 启动 8 卡 MoE 强化学习..."
echo " 注意: 每个 step 含 rollout 采样 + 梯度更新, 速度较慢"
echo " 注意: save_every=60, 每 60 步保存一次 checkpoint"
echo "============================================"
echo ""

# ===== 每卡日志: 输出到 <workspace>/logs/<run_id>_<random>/attempt_0/<rank>/{stdout,stderr}.log =====
LOG_DIR="${NANOCHAT_BASE_DIR}/logs"
mkdir -p "$LOG_DIR"
echo "  每卡日志目录: ${LOG_DIR}/"
echo ""

# ===== 启动 RL =====
# 加载 MoE SFT checkpoint (model-step 466), 超参对齐 dense run_wp4
# aux loss 默认保留 (正确接入, 不混入 logp); 剩余位置参数透传给 chat_rl
# (如 --no-aux-loss 关闭 aux 对比, --num-epochs 2 等)
/opt/venv/bin/python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=8 \
  --log-dir "$LOG_DIR" \
  --tee 3 \
  -m scripts.chat_rl \
  -- \
  --run=dummy \
  --model-step 466 \
  --device-batch-size=8 \
  --examples-per-step=16 \
  --num-samples=16 \
  --save-every=60 \
  "$@"
