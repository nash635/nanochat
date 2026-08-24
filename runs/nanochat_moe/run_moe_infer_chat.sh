#!/bin/bash
#
# Phase 6: MoE 推理基准测试 + CLI 对话演示
#
# 硬件: 8× NVIDIA H20-3e (Hopper SM 9.0, 96GB 每卡, 实际仅用 1 卡推理)
# 模型: 加载 MoE RL checkpoint (d24, step 466)
#   n_exp=8, top_k=2, stride=2 → 总参数 2.97B, active 1.61B
# 阶段:
#   1. infer_bench - 单卡推理基准 (延迟/吞吐/显存, 与 dense 同口径对比)
#   2. chat_cli    - 交互式对话 (带心算/数学题验证, 因 RL 为 GSM8K 数学方向)
#
# 前置条件:
#   - Phase 5 已完成 (cache_moe/chatrl_checkpoints/d24/ 有 model_000466.pt)
#   - cache_moe/tokenizer 已就绪 (symlink 共享 cache/tokenizer, 不重复下载)
#
# 用法:
#   bash runs/nanochat_moe/run_moe_infer_chat.sh
#
# 输出 (与 dense 完全隔离):
#   - 推理基准结果 (日志 /tmp/infer_bench_moe_rl.log)
#   - 交互式 CLI 对话
#
# 注意: 若 RL checkpoint 缺失会回退到 SFT (cache_moe/chatsft_checkpoints/d24/)。

set -euo pipefail

# 定位 nanochat 根目录（本脚本位于 runs/nanochat_moe/ 子目录）
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# ===== 环境配置 =====
# MoE 产物统一落到 cache_moe/，tokenizer 通过 symlink 共享 cache/ 的 (不重复下载)
export NANOCHAT_BASE_DIR="$ROOT_DIR/cache_moe"
export OMP_NUM_THREADS=1
export PATH=/opt/venv/bin:$PATH

CACHE_DIR="$ROOT_DIR/cache"

# ===== 建立隔离目录 + symlink 共享 tokenizer =====
mkdir -p "$NANOCHAT_BASE_DIR"
[ -e "$NANOCHAT_BASE_DIR/tokenizer" ] || ln -s "$CACHE_DIR/tokenizer" "$NANOCHAT_BASE_DIR/tokenizer"

echo "============================================"
echo " Phase 6: MoE 推理基准测试 + CLI 对话"
echo "============================================"
echo " 时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo " 硬件: 8× NVIDIA H20-3e (推理用 1 卡)"
echo " 模型: MoE RL checkpoint (d24, step 466, n_exp=8 top_k=2)"
echo "============================================"
echo ""

# ===== 环境检查 =====
echo "[1/3] 检查 Python 环境..."
/opt/venv/bin/python --version
/opt/venv/bin/python -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPUs: {torch.cuda.device_count()}')" 2>&1 | grep -v FutureWarning

echo "[2/3] 检查 checkpoint..."
RL_STEP=466
if ls -1 cache_moe/chatrl_checkpoints/d24/model_000466.pt 2>/dev/null; then
    echo "  [OK] RL checkpoint 就绪 (model_000466.pt)"
    SRC="rl"
elif ls -1 cache_moe/chatsft_checkpoints/d24/model_000466.pt 2>/dev/null; then
    echo "  [WARN] 无 RL checkpoint, 回退到 SFT (model_000466.pt)"
    SRC="sft"
else
    echo "  [FAIL] 无可用 checkpoint, 请先完成 Phase 4 (SFT) 或 Phase 5 (RL)"
    exit 1
fi

echo "[3/3] 检查 tokenizer..."
ls -1 cache_moe/tokenizer/tokenizer.pkl 2>/dev/null && echo "  [OK]" || {
    echo "  [FAIL] Tokenizer 缺失"
    exit 1
}

echo ""
echo "============================================"
echo " 阶段 1: 推理基准测试 (infer_bench)"
echo " 模型: $SRC checkpoint (step $RL_STEP)"
echo "============================================"
echo ""

/opt/venv/bin/python -m scripts.infer_bench \
  -i "$SRC" -s "$RL_STEP" 2>&1 | tee /tmp/infer_bench_moe_rl.log

echo ""
echo "============================================"
echo " 阶段 2: CLI 对话 (单轮测试)"
echo " 模型: $SRC checkpoint (step $RL_STEP)"
echo "============================================"
echo ""

# 数学题为主 (RL 为 GSM8K 数学方向), 另加通用知识题
PROMPTS=(
    "What is the capital of France?"
    "What is 24 * 37?"
    "If I have 12 apples and give away 4, then buy 8 more, how many do I have?"
    "What is the derivative of x^3 + 2x?"
)

for prompt in "${PROMPTS[@]}"; do
    echo ""
    echo "--------------------------------------------"
    echo " Q: $prompt"
    echo "--------------------------------------------"
    /opt/venv/bin/python -m scripts.chat_cli \
      -i "$SRC" -s "$RL_STEP" \
      -p "$prompt" \
      -t 0.6 2>&1
    echo ""
done

echo ""
echo "============================================"
echo " Phase 6 完成!"
echo " 推理基准日志: /tmp/infer_bench_moe_rl.log"
echo "============================================"
