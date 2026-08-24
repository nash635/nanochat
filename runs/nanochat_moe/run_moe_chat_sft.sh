#!/bin/bash
#
# Phase 4: MoE 监督微调 (SFT)
#
# 硬件: 8× NVIDIA H20-3e (Hopper SM 9.0, 96GB 每卡)
# 模型: 加载 MoE 基座 checkpoint (d24, step 12365, val_bpb=0.7450)
#   n_exp=8, top_k=2, stride=2 → 总参数 2.97B, active 1.61B
# 数据: 与 dense SFT 完全相同的 TaskMixture (SmolTalk + MMLU×3 + GSM8K×4)
#   (通过 symlink 共享 cache/ 的 task_data, 不重复下载 ~978MB)
# 优化器: 继承 MoE 基座优化器状态 (warm-start) + LR warmdown
#
# 前置条件:
#   - Phase 3 已完成 (cache_moe/base_checkpoints/d24/ 有 model_012365.pt)
#   - cache/task_data/ 已有 SFT 数据 (dense WP3 跑过即存在)
#   - /opt/venv/bin/python 可用
#
# 用法:
#   bash runs/nanochat_moe/run_moe_chat_sft.sh                  # 完整 SFT (默认 -1 = 全 epoch)
#   bash runs/nanochat_moe/run_moe_chat_sft.sh --num-iterations 4   # 冒烟测试 (透传任意 chat_sft 参数)
#
# 输出 (与 dense 完全隔离):
#   - Checkpoint: cache_moe/chatsft_checkpoints/d24/
#   - dense 的 cache/ 目录不受影响
#
# 注意: SFT 阶段 aux loss 仍然生效 (use_aux_loss=True), 因为路由稳定性对微调同样重要
#       (见 docs/nanochat_moe.md §2.3)。aux loss 已内建于 GPT.forward, 无需额外 flag。

set -euo pipefail

# 定位 nanochat 根目录（本脚本位于 runs/nanochat_moe/ 子目录）
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# ===== 环境配置 =====
# MoE 产物统一落到 cache_moe/，tokenizer/语料/SFT 数据通过 symlink 共享 cache/ 的 (不重复下载)
export NANOCHAT_BASE_DIR="$ROOT_DIR/cache_moe"
export NANOCHAT_DATASET_URL="https://hf-mirror.com"
export OMP_NUM_THREADS=1
export PATH=/opt/venv/bin:$PATH

CACHE_DIR="$ROOT_DIR/cache"

# ===== 建立隔离目录 + symlink 共享 tokenizer/语料/SFT 数据 =====
mkdir -p "$NANOCHAT_BASE_DIR"
[ -e "$NANOCHAT_BASE_DIR/tokenizer" ] || ln -s "$CACHE_DIR/tokenizer" "$NANOCHAT_BASE_DIR/tokenizer"
[ -e "$NANOCHAT_BASE_DIR/base_data_climbmix" ] || ln -s "$CACHE_DIR/base_data_climbmix" "$NANOCHAT_BASE_DIR/base_data_climbmix"
[ -e "$NANOCHAT_BASE_DIR/task_data" ] || ln -s "$CACHE_DIR/task_data" "$NANOCHAT_BASE_DIR/task_data"

# ===== 打印配置概览 =====
echo "============================================"
echo " Phase 4: MoE 监督微调 (SFT)"
echo "============================================"
echo " 时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo " 硬件: 8× NVIDIA H20-3e"
echo " 模型: 加载 MoE 基座 d24 (step 12365, val_bpb=0.7450)"
echo " 数据: SmolTalk + MMLU(x3) + GSM8K(x4)"
echo " Batch: 继承 MoE 基座配置 (device-batch-size=8)"
echo " Checkpoint 目录: cache_moe/chatsft_checkpoints/d24/"
echo "============================================"
echo ""

# ===== 环境检查 =====
echo "[1/4] 检查 Python 环境..."
/opt/venv/bin/python --version
/opt/venv/bin/python -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPUs: {torch.cuda.device_count()}')" 2>&1 | grep -v FutureWarning

echo "[2/4] 检查 GPU 可用性..."
nvidia-smi --query-gpu=index,name --format=csv,noheader 2>&1 | head -8

echo "[3/4] 检查 MoE 基座 checkpoint..."
ls -1 cache_moe/base_checkpoints/d24/model_012365.pt 2>/dev/null && echo "  [OK] 基座 checkpoint 就绪" || {
    echo "  [FAIL] 未找到 MoE 基座 checkpoint, 请先完成 Phase 3"
    exit 1
}
ls -1 cache/tokenizer/tokenizer.pkl 2>/dev/null && echo "  [OK] Tokenizer 就绪" || {
    echo "  [FAIL] Tokenizer 缺失, 请先完成 WP1"
    exit 1
}

echo "[4/4] 检查 SFT 数据 (symlink 共享 cache/task_data)..."
ls -1 cache/task_data/HuggingFaceTB--smol-smoltalk/*/*.parquet 2>/dev/null | head -1 >/dev/null && echo "  [OK] SmolTalk 就绪" || {
    echo "  [WARN] SmolTalk 缺失, 首次运行会从 hf-mirror.com 下载 (约几分钟)"
}

echo ""
echo "============================================"
echo " 启动 8 卡 MoE SFT..."
echo " 注意: SFT 仅在训练结束时保存 checkpoint (无 --save-every)"
echo "============================================"
echo ""

# ===== 每卡日志: 输出到 <workspace>/logs/<run_id>_<random>/attempt_0/<rank>/{stdout,stderr}.log =====
LOG_DIR="${NANOCHAT_BASE_DIR}/logs"
mkdir -p "$LOG_DIR"
echo "  每卡日志目录: ${LOG_DIR}/"
echo ""

# ===== 启动 SFT =====
# 加载 MoE base checkpoint (model-step 12365), 超参从 checkpoint meta 自动继承
# aux loss 已内建于 GPT.forward (use_aux_loss=True), 无需额外 flag
# 剩余位置参数透传给 chat_sft (如 --num-iterations 4 做冒烟测试)
/opt/venv/bin/python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=8 \
  --log-dir "$LOG_DIR" \
  --tee 3 \
  -m scripts.chat_sft \
  -- \
  --run=dummy \
  --model-step 12365 \
  "$@"
