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
        """Router returns (used_capacity, cb_weight, sec_mask) with correct shapes."""
        config = MockConfig()
        router = Router(config)
        B, T = 2, 8
        x = torch.randn(B, T, config.n_embd)
        used_cap, cb_weight, sec_mask = router(x)

        num_tokens = B * T
        exp_capacity = router.get_capacity(num_tokens)

        assert used_cap.shape == (config.n_exp,), f"used_capacity shape: {used_cap.shape}"
        assert cb_weight.shape == (num_tokens, config.n_exp, exp_capacity), \
            f"cb_weight shape: {cb_weight.shape}"
        assert sec_mask.shape == (num_tokens, config.n_exp, exp_capacity), \
            f"sec_mask shape: {sec_mask.shape}"

    def test_router_topk_indices(self):
        """Router selects exactly top_k experts per token."""
        config = MockConfig()
        router = Router(config)
        B, T = 1, 4
        x = torch.randn(B, T, config.n_embd)
        used_cap, cb_weight, sec_mask = router(x)

        # Count how many experts are selected per token
        num_tokens = B * T
        # sec_mask is [num_tokens, n_exp, exp_capacity] bool
        # Sum over capacity dim to get [num_tokens, n_exp]
        selected_per_token = sec_mask.view(num_tokens, config.n_exp, -1).sum(dim=-1)
        # Each token should select exactly top_k experts
        for i in range(num_tokens):
            assert selected_per_token[i].sum().item() == config.top_k, \
                f"Token {i} selected {selected_per_token[i].sum().item()} experts, expected {config.top_k}"

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
        """Expert capacity limits the number of tokens per expert."""
        config = MockConfig()
        config.min_capacity = 4  # small capacity for testing
        router = Router(config)
        router.train()  # use train_capacity
        B, T = 4, 16  # 64 tokens
        x = torch.randn(B, T, config.n_embd)
        used_cap, cb_weight, sec_mask = router(x)

        exp_capacity = router.get_capacity(B * T)
        # used_capacity should not exceed exp_capacity
        assert (used_cap <= exp_capacity).all(), \
            f"used_capacity exceeds exp_capacity: {used_cap} > {exp_capacity}"

    def test_noisy_top_k(self):
        """Noisy top-k router produces valid output."""
        config = MockConfig()
        config.use_noisy_top_k = True
        router = Router(config)
        B, T = 2, 8
        x = torch.randn(B, T, config.n_embd)
        used_cap, cb_weight, sec_mask = router(x)
        assert used_cap.shape == (config.n_exp,)
        assert cb_weight.shape[0] == B * T

    def test_router_deterministic(self):
        """Same input produces same output (no noise)."""
        config = MockConfig()
        config.use_noisy_top_k = False
        router = Router(config)
        x = torch.randn(2, 8, config.n_embd)
        out1 = router(x)
        out2 = router(x)
        for a, b in zip(out1, out2):
            torch.testing.assert_close(a, b)


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
        used_cap, cb_weight, sec_mask = router(x)
        # With n_exp=1 and top_k=1, all tokens go to the single expert
        assert used_cap.shape == (1,)
        assert sec_mask.shape[0] == B * T

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
