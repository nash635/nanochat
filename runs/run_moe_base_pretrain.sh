#!/bin/bash
#
# MoE 基座预训练 (Base Pretraining) — Phase 2/3
#
# 硬件: 8× NVIDIA H20-3e (Hopper SM 9.0, 96GB 每卡)
# 模型: 24层 Transformer, 每隔 stride=2 层插入一个 MoE 层 (12 MoE + 12 dense)
#   n_exp=8, top_k=2, stride=2 → 总参数 2.97B, active 1.61B (54.2%, 含 value_embeds 0.60B 常驻)
# 数据: 复用 dense 的 ClimbMix 语料 + 32768 tokenizer (symlink, 不重复下载)
# 优化器: DistMuonAdamW (expert 3D 参数归 AdamW) + 可选 FP8 tensorwise (expert/router 层跳过)
#
# 前置条件:
#   - WP1 已完成 (tokenizer + 171 shard 语料在 cache/ 下)
#   - /opt/venv/bin/python 可用且已装 rustbpe (脚本会自动检查)
#
# 用法:
#   bash runs/run_moe_base_pretrain.sh            # FP8 版 (默认, tensorwise scaling)
#   bash runs/run_moe_base_pretrain.sh nofp8      # 无 FP8 (BF16), 用于二分验证 loss 发散是否由 FP8×MoE 引起
#
# 输出 (与 dense 完全隔离):
#   - cache_moe/base_checkpoints/d24/   (FP8 与 noFP8 共用同一目录)
#   - dense 的 cache/ 目录不受影响

set -euo pipefail

# ===== 模式开关 =====
MODE="${1:-fp8}"
case "$MODE" in
  fp8)   USE_FP8=1 ;;
  nofp8) USE_FP8=0 ;;
  *)
    echo "用法: $0 [fp8|nofp8]"
    echo "  fp8   默认, 启用 FP8 tensorwise (expert/router 层跳过)"
    echo "  nofp8 全程 BF16, 排除 FP8 数值稳定性嫌疑"
    exit 1
    ;;
esac

# ===== 环境配置 =====
# MoE 产物统一落到 cache_moe/，tokenizer 与语料通过 symlink 共享 cache/ 的 (不重复下载)
export NANOCHAT_BASE_DIR="/volume/posttrain/users/lqiu/src/nanochat/cache_moe"
export NANOCHAT_DATASET_URL="https://hf-mirror.com"
export HF_ENDPOINT="https://hf-mirror.com"   # FA3 kernel (varunneal) 也走镜像下载
export OMP_NUM_THREADS=1

# ===== NCCL 诊断 (用于定位集合通信 hang) =====
# 挂掉时 dump 每个 rank 的通信轨迹 + 精确打印哪个 rank/seq 失同步
export NCCL_DEBUG=INFO
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000
export TORCH_NCCL_BLOCKING_WAIT=1   # 超时后立刻抛错, 不干等 600s

cd /volume/posttrain/users/lqiu/src/nanochat
export PATH=/opt/venv/bin:$PATH

CACHE_DIR="/volume/posttrain/users/lqiu/src/nanochat/cache"

# ===== 建立隔离目录 + symlink 共享 tokenizer/语料 =====
mkdir -p "$NANOCHAT_BASE_DIR"
[ -e "$NANOCHAT_BASE_DIR/tokenizer" ] || ln -s "$CACHE_DIR/tokenizer" "$NANOCHAT_BASE_DIR/tokenizer"
[ -e "$NANOCHAT_BASE_DIR/base_data_climbmix" ] || ln -s "$CACHE_DIR/base_data_climbmix" "$NANOCHAT_BASE_DIR/base_data_climbmix"

# ===== 打印配置概览 =====
if [ "$USE_FP8" = "1" ]; then
  MODE_LABEL="FP8 tensorwise"
  MODE_MODEL="--fp8"
else
  MODE_LABEL="无 FP8 (BF16)"
  MODE_MODEL="(无 --fp8)"
fi
echo "============================================"
echo " MoE 基座预训练 (Phase 2/3) — ${MODE_LABEL}"
echo "============================================"
echo " 时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo " 硬件: 8× NVIDIA H20-3e"
echo " 模型: --depth=24 --n-exp=8 --top-k=2 --stride=2 ${MODE_MODEL}"
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
echo " 启动 8 卡 MoE 预训练 (${MODE_LABEL})..."
echo "============================================"
echo ""

# ===== 每卡日志: 输出到 <workspace>/logs/<run_id>_<random>/attempt_0/<rank>/{stdout,stderr}.log =====
# --tee 3 = stdout+stderr 同时写入文件与控制台 (控制台仍是 8 卡交错汇总, 文件按 rank 隔离)
LOG_DIR="${NANOCHAT_BASE_DIR}/logs"
mkdir -p "$LOG_DIR"
echo "  每卡日志目录: ${LOG_DIR}/"
echo ""

# ===== 组装训练参数 =====
# MoE 超参对齐 docs/nanochat_moe.md §2.2:
#   n_exp=8, top_k=2, stride=2, aux_loss_weight=0.01, router_z_loss_weight=0.001
#   train_capacity=1.25, eval_capacity=2.0, min_capacity=4, router_use_full_prec=True
# target-param-data-ratio=5.6: 按 active 参数 (1.61B) × Chinchilla 8 ≈ 12.9B tokens
#   (代码用 scaling params 2.315B 计算: 5.6 × 2.315B = 12.97B → num_iterations ≈ 12,365)
# --grad-clip=1.0: ZeRO-2 全局梯度裁剪安全网
TRAIN_ARGS=(
  --depth=24
  --target-param-data-ratio=5.6
  --device-batch-size=8
  --save-every=500
  --grad-clip=1.0
  --n-exp=8
  --top-k=2
  --stride=2
  --use-aux-loss
  --aux-loss-weight=0.01
  --use-router-z-loss
  --router-z-loss-weight=0.001
  --train-capacity=1.25
  --eval-capacity=2.0
  --min-capacity=4
  --use-switch-tfm-init
  --router-use-full-prec
  --run=dummy
)
if [ "$USE_FP8" = "1" ]; then
  TRAIN_ARGS+=(--fp8)
fi

# ===== 启动训练 =====
/opt/venv/bin/python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=8 \
  --log-dir "$LOG_DIR" \
  --tee 3 \
  -m scripts.base_train \
  -- \
  "${TRAIN_ARGS[@]}"
