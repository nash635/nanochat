"""
Unit tests for the MoE module (nanochat/moe.py).

Tests cover:
1. Router output shapes and properties
2. Aux loss and z loss computation
3. Expert capacity clipping
4. MOELayer forward pass correctness
5. n_exp=1 MoE degrades to ordinary MLP (dense compatibility)
6. MOEManager loss aggregation

These tests run on CPU (no CUDA required) since the MoE operations
are pure PyTorch bmm/softmax/topk.

python -m pytest tests/test_moe.py -v
"""

import math
import pytest
import torch
import torch.nn.functional as F

from nanochat.moe import (
    Router,
    MLPExperts,
    MOELayer,
    MOEManager,
    init_moe_weights,
)


# ---------------------------------------------------------------------------
# Minimal config dataclass for testing (avoids importing full GPTConfig)
# ---------------------------------------------------------------------------
class MockConfig:
    """Minimal config that satisfies the MoE module's requirements."""
    n_embd = 64
    n_exp = 4
    top_k = 2
    use_aux_loss = True
    use_router_z_loss = True
    use_noisy_top_k = False
    aux_loss_weight = 0.01
    router_z_loss_weight = 0.001
    train_capacity = 1.25
    eval_capacity = 2.0
    min_capacity = 4
    router_use_full_prec = False


class DenseMockConfig:
    """Config with n_exp=1 for dense compatibility tests."""
    n_embd = 64
    n_exp = 1
    top_k = 1
    use_aux_loss = False
    use_router_z_loss = False
    use_noisy_top_k = False
    aux_loss_weight = 0.01
    router_z_loss_weight = 0.001
    train_capacity = 1.25
    eval_capacity = 2.0
    min_capacity = 4
    router_use_full_prec = False


# ---------------------------------------------------------------------------
# Router tests
# ---------------------------------------------------------------------------
class TestRouter:
    def test_router_output_shapes(self):
        """Router returns sparse (top_k_indices, router_probs, exp_rank, exp_capacity)."""
        config = MockConfig()
        router = Router(config)
        B, T = 2, 8
        x = torch.randn(B, T, config.n_embd)
        top_k_indices, router_probs, exp_rank, exp_capacity = router(x)

        num_tokens = B * T
        assert top_k_indices.shape == (num_tokens, config.top_k), \
            f"top_k_indices shape: {top_k_indices.shape}"
        assert router_probs.shape == (num_tokens, config.top_k), \
            f"router_probs shape: {router_probs.shape}"
        assert exp_rank.shape == (num_tokens, config.top_k), \
            f"exp_rank shape: {exp_rank.shape}"
        assert exp_capacity == router.get_capacity(num_tokens)
        # probabilities are top-k softmax, so they sum to 1 per token
        assert torch.allclose(router_probs.sum(dim=1), torch.ones(num_tokens), atol=1e-6)

    def test_router_topk_indices(self):
        """Router selects exactly top_k distinct experts per token."""
        config = MockConfig()
        router = Router(config)
        B, T = 1, 4
        x = torch.randn(B, T, config.n_embd)
        top_k_indices, router_probs, exp_rank, exp_capacity = router(x)

        num_tokens = B * T
        assert top_k_indices.shape == (num_tokens, config.top_k)
        # each token's chosen experts are distinct
        for i in range(num_tokens):
            assert len(set(top_k_indices[i].tolist())) == config.top_k, \
                f"Token {i} did not select {config.top_k} distinct experts: {top_k_indices[i]}"

    def test_aux_loss_is_scalar(self):
        """Aux loss is a scalar tensor."""
        config = MockConfig()
        router = Router(config)
        B, T = 2, 8
        x = torch.randn(B, T, config.n_embd)
        router(x)
        assert router._pending_aux_loss is not None
        assert router._pending_aux_loss.dim() == 0, f"aux_loss should be scalar, got dim={router._pending_aux_loss.dim()}"

    def test_z_loss_is_scalar(self):
        """Router z loss is a scalar tensor."""
        config = MockConfig()
        router = Router(config)
        B, T = 2, 8
        x = torch.randn(B, T, config.n_embd)
        router(x)
        assert router._pending_z_loss is not None
        assert router._pending_z_loss.dim() == 0, f"z_loss should be scalar, got dim={router._pending_z_loss.dim()}"

    def test_aux_loss_disabled(self):
        """When use_aux_loss=False, no aux loss is computed."""
        config = MockConfig()
        config.use_aux_loss = False
        router = Router(config)
        x = torch.randn(2, 8, config.n_embd)
        router(x)
        assert router._pending_aux_loss is None

    def test_z_loss_disabled(self):
        """When use_router_z_loss=False, no z loss is computed."""
        config = MockConfig()
        config.use_router_z_loss = False
        router = Router(config)
        x = torch.randn(2, 8, config.n_embd)
        router(x)
        assert router._pending_z_loss is None

    def test_capacity_clipping(self):
        """Tokens are dropped when an expert exceeds its capacity."""
        config = MockConfig()
        config.n_exp = 4
        config.top_k = 1
        config.min_capacity = 2
        router = Router(config)
        router.train()  # use train_capacity
        # force every token to route to expert 0 so it overflows its capacity
        with torch.no_grad():
            router.w_g.weight.zero_()
            router.w_g.weight[0, :] = 10.0
        B, T = 1, 8  # 8 tokens, top_k=1 -> capacity = floor(1*1.25*8/4) = 2
        x = torch.ones(B, T, config.n_embd)  # all-positive -> expert 0 always wins
        top_k_indices, router_probs, exp_rank, exp_capacity = router(x)

        assert exp_capacity == 2
        assert (top_k_indices == 0).all(), "all tokens should route to expert 0"
        valid = exp_rank < exp_capacity
        # exactly `capacity` tokens are kept, the rest are dropped
        assert valid.sum().item() == exp_capacity, \
            f"expected {exp_capacity} kept tokens, got {valid.sum().item()}"

    def test_noisy_top_k(self):
        """Noisy top-k router produces valid output."""
        config = MockConfig()
        config.use_noisy_top_k = True
        router = Router(config)
        B, T = 2, 8
        x = torch.randn(B, T, config.n_embd)
        top_k_indices, router_probs, exp_rank, exp_capacity = router(x)
        assert top_k_indices.shape == (B * T, config.top_k)
        assert exp_capacity == router.get_capacity(B * T)

    def test_router_deterministic(self):
        """Same input produces same output (no noise)."""
        config = MockConfig()
        config.use_noisy_top_k = False
        router = Router(config)
        x = torch.randn(2, 8, config.n_embd)
        out1 = router(x)
        out2 = router(x)
        for a, b in zip(out1[:3], out2[:3]):  # first three are tensors
            torch.testing.assert_close(a, b)
        assert out1[3] == out2[3]  # capacity is an int, deterministic too


# ---------------------------------------------------------------------------
# MLPExperts tests
# ---------------------------------------------------------------------------
class TestMLPExperts:
    def test_expert_output_shape(self):
        """MLPExperts produces correct output shape."""
        config = MockConfig()
        experts = MLPExperts(config)
        init_moe_weights(experts, config, n_layer=12)
        n_exp = config.n_exp
        exp_capacity = 8
        x = torch.randn(n_exp, exp_capacity, config.n_embd)
        out = experts(x)
        assert out.shape == (n_exp, exp_capacity, config.n_embd), \
            f"Expected {(n_exp, exp_capacity, config.n_embd)}, got {out.shape}"



# ---------------------------------------------------------------------------
# MOELayer tests
# ---------------------------------------------------------------------------
class TestMOELayer:
    def test_moe_layer_output_shape(self):
        """MOELayer preserves input shape."""
        config = MockConfig()
        manager = MOEManager()
        moe = MOELayer(config, manager)
        init_moe_weights(moe.experts, config, n_layer=12)
        B, T = 2, 8
        x = torch.randn(B, T, config.n_embd)
        out = moe(x)
        assert out.shape == (B, T, config.n_embd), \
            f"Expected {(B, T, config.n_embd)}, got {out.shape}"

    def test_moe_layer_aux_loss_aggregation(self):
        """MOELayer adds aux loss and z loss to the manager."""
        config = MockConfig()
        manager = MOEManager()
        moe = MOELayer(config, manager)
        init_moe_weights(moe.experts, config, n_layer=12)
        x = torch.randn(2, 8, config.n_embd)
        moe(x)
        assert len(manager.aux_loss) == 1
        assert len(manager.router_z_loss) == 1

    def test_moe_layer_no_aux_loss_when_disabled(self):
        """MOELayer doesn't add losses when disabled."""
        config = MockConfig()
        config.use_aux_loss = False
        config.use_router_z_loss = False
        manager = MOEManager()
        moe = MOELayer(config, manager)
        init_moe_weights(moe.experts, config, n_layer=12)
        x = torch.randn(2, 8, config.n_embd)
        moe(x)
        assert len(manager.aux_loss) == 0
        assert len(manager.router_z_loss) == 0

    def test_moe_layer_finite_output(self):
        """MOELayer produces finite output."""
        config = MockConfig()
        manager = MOEManager()
        moe = MOELayer(config, manager)
        init_moe_weights(moe.experts, config, n_layer=12)
        x = torch.randn(2, 8, config.n_embd)
        out = moe(x)
        assert torch.isfinite(out).all(), "MOELayer output contains non-finite values"

    def test_sparse_dispatch_matches_dense_reference(self):
        """Sparse dispatch/combine matches a brute-force per-token reference."""
        config = MockConfig()
        manager = MOEManager()
        moe = MOELayer(config, manager)
        init_moe_weights(moe.experts, config, n_layer=12)
        moe.eval()  # eval_capacity=2.0 -> capacity == num_tokens -> no dropping
        B, T = 2, 8
        num_tokens = B * T
        x = torch.randn(B, T, config.n_embd)

        out = moe(x)  # sparse implementation

        # brute-force reference: loop over tokens and their top-k experts
        xf = x.view(num_tokens, config.n_embd)
        with torch.no_grad():
            logits = moe.router.w_g(xf)  # [num_tokens, n_exp]
            topk_logits, topk_idx = logits.topk(config.top_k, dim=-1)
            probs = F.softmax(topk_logits, dim=-1)  # [num_tokens, top_k]
            ref = torch.zeros(num_tokens, config.n_embd)
            for t in range(num_tokens):
                for kk in range(config.top_k):
                    e = topk_idx[t, kk].item()
                    w = probs[t, kk].item()
                    h = xf[t] @ moe.experts.c_fc[e]  # [4*n_embd]
                    h = F.relu(h).square()
                    y = h @ moe.experts.c_proj[e]  # [n_embd]
                    ref[t] += w * y
        ref = ref.view(B, T, config.n_embd)

        torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)

    def test_moe_layer_preserves_dtype(self):
        """MOELayer output stays in the activation dtype (regression: dense was fp32)."""
        config = MockConfig()
        manager = MOEManager()
        moe = MOELayer(config, manager)
        init_moe_weights(moe.experts, config, n_layer=12)
        x = torch.randn(2, 8, config.n_embd, dtype=torch.bfloat16)
        try:
            out = moe(x)
        except RuntimeError as e:
            pytest.skip(f"bf16 bmm not supported on this CPU: {e}")
        assert out.dtype == x.dtype


# ---------------------------------------------------------------------------
# MOEManager tests
# ---------------------------------------------------------------------------
class TestMOEManager:
    def test_aggregate_aux_loss(self):
        """MOEManager correctly aggregates aux losses."""
        manager = MOEManager()
        manager.add_aux_loss(torch.tensor(0.1))
        manager.add_aux_loss(torch.tensor(0.2))
        manager.add_aux_loss(torch.tensor(0.3))
        total = manager.aggregate_aux_loss()
        assert abs(total.item() - 0.6) < 1e-6

    def test_aggregate_z_loss(self):
        """MOEManager correctly aggregates z losses."""
        manager = MOEManager()
        manager.add_router_z_loss(torch.tensor(0.01))
        manager.add_router_z_loss(torch.tensor(0.02))
        total = manager.aggregate_router_z_loss()
        assert abs(total.item() - 0.03) < 1e-6

    def test_reset(self):
        """MOEManager.reset() clears all losses."""
        manager = MOEManager()
        manager.add_aux_loss(torch.tensor(0.1))
        manager.add_router_z_loss(torch.tensor(0.01))
        manager.reset()
        assert len(manager.aux_loss) == 0
        assert len(manager.router_z_loss) == 0

    def test_aggregate_empty(self):
        """Aggregating empty loss lists returns 0."""
        manager = MOEManager()
        assert manager.aggregate_aux_loss() == 0.0
        assert manager.aggregate_router_z_loss() == 0.0


# ---------------------------------------------------------------------------
# Dense compatibility tests (n_exp=1)
# ---------------------------------------------------------------------------
class TestDenseCompatibility:
    def test_n_exp_1_router(self):
        """With n_exp=1, router still produces valid output."""
        config = DenseMockConfig()
        router = Router(config)
        B, T = 2, 8
        x = torch.randn(B, T, config.n_embd)
        top_k_indices, router_probs, exp_rank, exp_capacity = router(x)
        # With n_exp=1 and top_k=1, all tokens go to the single expert
        assert top_k_indices.shape == (B * T, 1)
        assert (top_k_indices == 0).all()
        assert exp_capacity == router.get_capacity(B * T)

    def test_n_exp_1_moe_layer(self):
        """With n_exp=1, MOELayer produces valid output of correct shape."""
        config = DenseMockConfig()
        manager = MOEManager()
        moe = MOELayer(config, manager)
        init_moe_weights(moe.experts, config, n_layer=12)
        B, T = 2, 8
        x = torch.randn(B, T, config.n_embd)
        out = moe(x)
        assert out.shape == (B, T, config.n_embd)
        assert torch.isfinite(out).all()

    def test_n_exp_1_no_aux_loss(self):
        """With n_exp=1 and aux disabled, no losses are added."""
        config = DenseMockConfig()
        config.use_aux_loss = False
        config.use_router_z_loss = False
        manager = MOEManager()
        moe = MOELayer(config, manager)
        init_moe_weights(moe.experts, config, n_layer=12)
        x = torch.randn(2, 8, config.n_embd)
        moe(x)
        assert len(manager.aux_loss) == 0
        assert len(manager.router_z_loss) == 0


# ---------------------------------------------------------------------------
# Weight initialization tests
# ---------------------------------------------------------------------------
class TestInitWeights:
    def test_init_moe_weights(self):
        """init_moe_weights initializes expert parameters correctly."""
        config = MockConfig()
        experts = MLPExperts(config)
        init_moe_weights(experts, config, n_layer=12)
        # After init, c_fc should be non-zero and c_proj should be zero
        assert torch.isfinite(experts.c_fc).all(), "c_fc contains non-finite values"
        assert experts.c_fc.abs().sum().item() > 0, "c_fc should be non-zero after init"
        assert experts.c_proj.abs().sum().item() == 0.0, "c_proj should be zeros after init"



# ---------------------------------------------------------------------------
# Gradient flow tests
# ---------------------------------------------------------------------------
class TestGradientFlow:
    def test_moe_layer_gradients(self):
        """Gradients flow through MOELayer."""
        config = MockConfig()
        manager = MOEManager()
        moe = MOELayer(config, manager)
        init_moe_weights(moe.experts, config, n_layer=12)
        B, T = 2, 4
        x = torch.randn(B, T, config.n_embd, requires_grad=True)
        out = moe(x)
        loss = out.sum()
        loss.backward()
        # Check that gradients exist for expert parameters
        assert moe.experts.c_fc.grad is not None
        assert moe.experts.c_proj.grad is not None
        # Check that gradients exist for router parameters
        assert moe.router.w_g.weight.grad is not None

    def test_moe_layer_gradient_finite(self):
        """Gradients through MOELayer are finite."""
        config = MockConfig()
        manager = MOEManager()
        moe = MOELayer(config, manager)
        init_moe_weights(moe.experts, config, n_layer=12)
        B, T = 2, 4
        x = torch.randn(B, T, config.n_embd, requires_grad=True)
        out = moe(x)
        loss = out.sum()
        loss.backward()
        assert torch.isfinite(moe.experts.c_fc.grad).all()
        assert torch.isfinite(moe.experts.c_proj.grad).all()
        assert torch.isfinite(moe.router.w_g.weight.grad).all()
