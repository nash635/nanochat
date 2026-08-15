#!/bin/bash
#
# MoE 基座预训练 (Base Pretraining) — Phase 2/3
#
# 硬件: 8× NVIDIA H20-3e (Hopper SM 9.0, 96GB 每卡)
# 模型: 24层 Transformer, 每隔 stride=2 层插入一个 MoE 层 (12 MoE + 12 dense)
#   n_exp=8, top_k=2, stride=2 → 总参数 2.97B, active 1.61B (54.2%, 含 value_embeds 0.60B 常驻)
# 数据: 复用 dense 的 ClimbMix 语料 + 32768 tokenizer (symlink, 不重复下载)
# 优化器: DistMuonAdamW (expert 3D 参数归 AdamW) + FP8 tensorwise (expert/router 层跳过)
#
# 前置条件:
#   - WP1 已完成 (tokenizer + 171 shard 语料在 cache/ 下)
#   - /opt/venv/bin/python 可用且已装 rustbpe (脚本会自动检查)
#
# 用法:
#   bash runs/run_moe_base_pretrain.sh
#
# 输出 (与 dense 完全隔离):
#   - Checkpoint: cache_moe/base_checkpoints/d24/
#   - dense 的 cache/ 目录不受影响

set -euo pipefail

# ===== 环境配置 =====
# 隔离 MoE 产物到 cache_moe/，tokenizer 与语料通过 symlink 共享 cache/ 的 (不重复下载)
export NANOCHAT_BASE_DIR="/volume/posttrain/users/lqiu/src/nanochat/cache_moe"
export NANOCHAT_DATASET_URL="https://hf-mirror.com"
export HF_ENDPOINT="https://hf-mirror.com"   # FA3 kernel (varunneal) 也走镜像下载
export OMP_NUM_THREADS=1
cd /volume/posttrain/users/lqiu/src/nanochat
export PATH=/opt/venv/bin:$PATH

CACHE_DIR="/volume/posttrain/users/lqiu/src/nanochat/cache"

# ===== 建立 cache_moe 隔离目录 + symlink 共享 tokenizer/语料 =====
mkdir -p "$NANOCHAT_BASE_DIR"
[ -e "$NANOCHAT_BASE_DIR/tokenizer" ] || ln -s "$CACHE_DIR/tokenizer" "$NANOCHAT_BASE_DIR/tokenizer"
[ -e "$NANOCHAT_BASE_DIR/base_data_climbmix" ] || ln -s "$CACHE_DIR/base_data_climbmix" "$NANOCHAT_BASE_DIR/base_data_climbmix"

# ===== 打印配置概览 =====
echo "============================================"
echo " MoE 基座预训练 (Phase 2/3)"
echo "============================================"
echo " 时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo " 硬件: 8× NVIDIA H20-3e"
echo " 模型: --depth=24 --n-exp=8 --top-k=2 --stride=2 --fp8"
echo " 数据: --target-param-data-ratio=5.6 (active 口径 Chinchilla ≈12.9B tokens)"
echo " Batch: --device-batch-size=8, 每 500 步保存 checkpoint"
echo " Checkpoint 目录: cache_moe/base_checkpoints/d24/"
echo "============================================"
echo ""

# ===== 环境检查 =====
echo "[1/4] 检查 Python 环境..."
/opt/venv/bin/python --version
/opt/venv/bin/python -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPUs: {torch.cuda.device_count()}')" 2>&1 | grep -v FutureWarning

echo "[2/4] 检查 GPU 可用性..."
nvidia-smi --query-gpu=index,name --format=csv,noheader 2>&1 | head -8

echo "[3/4] 检查 tokenizer 和数据 (通过 symlink 共享 cache/)..."
ls -1 cache/tokenizer/tokenizer.pkl cache/tokenizer/token_bytes.pt 2>/dev/null && echo "  [OK] Tokenizer" || { echo "  [FAIL] Tokenizer 缺失, 请先完成 WP1"; exit 1; }
SHARD_COUNT=$(ls cache/base_data_climbmix/*.parquet 2>/dev/null | wc -l)
echo "  数据 shard: ${SHARD_COUNT}/171"
[ "$SHARD_COUNT" -ge 1 ] && echo "  [OK] 数据就绪" || { echo "  [FAIL] 数据缺失, 请先完成 WP1"; exit 1; }

echo "[4/4] 检查 rustbpe..."
/opt/venv/bin/python -c "import rustbpe" 2>/dev/null && echo "  [OK] rustbpe" || {
    echo "  [WARN] rustbpe 未安装, 正在安装..."
    pip install rustbpe --index-url https://pypi.org/simple/ 2>&1 | tail -1
}

echo "[5/5] 检查 kernels (FA3 依赖, 缺失会导致训练慢 ~10x)..."
/opt/venv/bin/python -c "from nanochat.flash_attention import USE_FA3; assert USE_FA3" 2>/dev/null && echo "  [OK] FA3 可用" || {
    echo "  [WARN] FA3 不可用 (kernels 缺失), 正在安装 kernels..."
    pip install "kernels>=0.11.7" 2>&1 | tail -1
    /opt/venv/bin/python -c "from nanochat.flash_attention import USE_FA3; assert USE_FA3" 2>/dev/null && echo "  [OK] FA3 已恢复" || echo "  [WARN] 仍不可用, 训练将非常慢 (SDPA fallback)"
}

echo ""
echo "============================================"
echo " 启动 8 卡 MoE 预训练..."
echo "============================================"
echo ""

# ===== 启动训练 =====
# MoE 超参对齐 docs/nanochat_moe.md §2.2:
#   n_exp=8, top_k=2, stride=2, aux_loss_weight=0.01, router_z_loss_weight=0.001
#   train_capacity=1.25, eval_capacity=2.0, min_capacity=4, router_use_full_prec=True
# target-param-data-ratio=5.6: 按 active 参数 (1.61B) × Chinchilla 8 ≈ 12.9B tokens
#   (代码用 scaling params 2.315B 计算: 5.6 × 2.315B = 12.97B → num_iterations ≈ 12,365)
/opt/venv/bin/python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=8 \
  -m scripts.base_train \
  -- \
  --depth=24 \
  --target-param-data-ratio=5.6 \
  --device-batch-size=8 \
  --save-every=500 \
  --fp8 \
  --n-exp=8 \
  --top-k=2 \
  --stride=2 \
  --use-aux-loss \
  --aux-loss-weight=0.01 \
  --use-router-z-loss \
  --router-z-loss-weight=0.001 \
  --train-capacity=1.25 \
  --eval-capacity=2.0 \
  --min-capacity=4 \
  --use-switch-tfm-init \
  --router-use-full-prec \
  --run=dummy
