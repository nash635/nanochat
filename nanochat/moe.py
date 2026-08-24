"""
MoE (Mixture of Experts) module for nanochat.

Ports the Router, MLPExperts, and MOELayer from nanoMoE, adapted to:
- nanochat's Linear class (casts weights to input dtype in forward)
- nanoMoE's GELU activation (F.gelu(x)) — relu² was replaced because under the
  AdamW expert optimizer it grows unboundedly, unlike nanochat's dense MLP whose
  relu² is stabilized by the Muon spectral-norm bound.
- nanochat's GPTConfig dataclass

The MOELayer replaces the standard MLP in every `stride`-th block,
keeping all other nanochat features (RoPE, QK norm, FA3, etc.) intact.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Linear):
    """nn.Linear that casts weights to match input dtype in forward.
    Same as nanochat.gpt.Linear, defined locally to avoid circular import."""
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))


class MOEManager:
    """
    Tracks, stores, and aggregates auxiliary losses across multiple MoE layers.
    One instance per model, passed to each MOELayer.
    """

    def __init__(self):
        self.aux_loss = []
        self.router_z_loss = []
        self.expert_loads = []
        # Cached aggregates from the last forward pass, so the training loop can
        # log router diagnostics (aux/z loss magnitude and per-expert load) after
        # `reset()` has already cleared the per-layer lists.
        self.last_aux_loss = None
        self.last_router_z_loss = None
        self.last_expert_load = None

    def reset(self):
        self.aux_loss = []
        self.router_z_loss = []
        self.expert_loads = []

    def add_aux_loss(self, loss):
        self.aux_loss.append(loss)

    def add_router_z_loss(self, loss):
        self.router_z_loss.append(loss)

    def add_expert_load(self, load):
        self.expert_loads.append(load)

    def aggregate_aux_loss(self):
        self.last_aux_loss = sum(self.aux_loss)
        return self.last_aux_loss

    def aggregate_router_z_loss(self):
        self.last_router_z_loss = sum(self.router_z_loss)
        return self.last_router_z_loss

    def aggregate_expert_load(self):
        """Sum per-layer expert token counts into a single [n_exp] tensor (for logging)."""
        if not self.expert_loads:
            self.last_expert_load = None
            return None
        self.last_expert_load = torch.stack(self.expert_loads).sum(dim=0)
        return self.last_expert_load


class Router(nn.Module):
    """
    Noisy top-k router with Switch Transformer auxiliary loss and ST-MoE z loss.

    Adapted from nanoMoE Router to use nanochat's Linear class.
    """

    def __init__(self, config):
        super().__init__()

        # router settings
        self.top_k = config.top_k
        self.n_exp = config.n_exp
        assert self.top_k >= 1 and self.top_k <= config.n_exp
        self.use_noisy_top_k = config.use_noisy_top_k
        self.train_capacity = config.train_capacity
        self.eval_capacity = config.eval_capacity
        self.min_capacity = config.min_capacity
        self.router_use_full_prec = config.router_use_full_prec

        # auxiliary / load balancing loss settings
        self.use_aux_loss = config.use_aux_loss
        self.use_router_z_loss = config.use_router_z_loss

        # linear projection for (noisy) softmax gating
        # no bias is used, see page 4 eq (4) in https://arxiv.org/abs/1701.06538
        self.w_g = Linear(config.n_embd, config.n_exp, bias=False)
        self.w_noise = Linear(config.n_embd, config.n_exp, bias=False) if self.use_noisy_top_k else None

    def forward(self, x):
        # optionally run the router in full precision to avoid instability during training
        # see discussion on pg. 9 here: https://arxiv.org/abs/2101.03961
        # setting enabled to False in autocast automatically puts everything in float32
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.router_use_full_prec:
            ctx = torch.amp.autocast(device_type=device_type, enabled=False)
        else:
            from contextlib import nullcontext
            ctx = nullcontext()

        with ctx:
            B, T, _ = x.size()
            num_tokens = B * T

            # eq (4) in https://arxiv.org/abs/1701.06538
            logits = self.w_g(x)  # [B, T, n_exp]
            if self.use_noisy_top_k:
                # optionally add noise into the router
                noise = F.softplus(self.w_noise(x))
                noise *= torch.randn_like(noise)
                logits += noise

            # router z loss, computed on logits (before softmax)
            # this loss prevents router logits from becoming too large
            if self.use_router_z_loss:
                z_loss = self.compute_router_z_loss(logits)
                self._pending_z_loss = z_loss
            else:
                self._pending_z_loss = None

            # find top k experts for each token
            top_k_logits, top_k_indices = logits.topk(self.top_k, dim=-1)  # [B, T, k]
            top_k_indices = top_k_indices.view(num_tokens, self.top_k)  # [num_tokens, k]

            # record per-expert token load for diagnostics (no grad, training only).
            # counts every (token, choice) pair, i.e. each expert's routed-token count
            # before capacity dropping — the primary signal for expert collapse.
            self._pending_expert_load = (
                torch.bincount(top_k_indices.view(-1), minlength=self.n_exp).float()
                if self.training else None
            )

            # normalize expert probabilities over top-k. Softmax over just the top-k
            # logits equals softmax over the full vector with -inf on the rest (those
            # contribute 0 weight), so this is equivalent to the dense version.
            router_probs = F.softmax(top_k_logits, dim=-1).view(num_tokens, self.top_k)  # [num_tokens, k]

            # compute auxiliary load balancing loss
            # this loss encourages equal probability assigned to each expert
            # and equal load balancing of tokens assigned to each expert
            if self.use_aux_loss:
                # reconstruct the full [num_tokens, n_exp] prob tensor (zeros off top-k)
                full_probs = torch.zeros(num_tokens, self.n_exp, device=logits.device, dtype=router_probs.dtype)
                full_probs.scatter_(1, top_k_indices, router_probs)
                self._pending_aux_loss = self.compute_aux_loss(full_probs, top_k_indices)
            else:
                self._pending_aux_loss = None

            # compute expert capacity
            exp_capacity = self.get_capacity(num_tokens)

            # rank each token within its assigned expert, in the same order as the
            # original dense cumsum: all top-1 choices first (token order), then top-2, etc.
            exp_rank = self._compute_exp_rank(top_k_indices)  # [num_tokens, k]

            # Return sparse routing info instead of the dense [num_tokens, n_exp, capacity]
            # tensors (which cost O(tokens * capacity) memory). MOELayer uses these to
            # gather/scatter directly in O(tokens * k).
            return top_k_indices, router_probs, exp_rank, exp_capacity

    def _compute_exp_rank(self, top_k_indices):
        """
        Rank of each token within its assigned expert, matching the dense cumsum order.

        Args:
            top_k_indices: [num_tokens, k] long tensor of chosen expert ids.

        Returns:
            [num_tokens, k] long tensor; entry (t, i) is the rank (0-based) of token t
            within the expert it selected as its i-th choice. Tokens whose rank is
            >= capacity are dropped later in MOELayer.forward.
        """
        num_tokens, k = top_k_indices.shape
        device = top_k_indices.device
        pos = torch.arange(num_tokens, dtype=torch.long, device=device)
        ranks = []
        offset = torch.zeros(self.n_exp, dtype=torch.long, device=device)
        for i in range(k):
            e = top_k_indices[:, i]  # [num_tokens]
            # cumulative count of each expert over the token order -> rank within expert
            cnt = torch.cumsum(F.one_hot(e, num_classes=self.n_exp), dim=0) - 1  # [num_tokens, n_exp]
            within = cnt[pos, e]  # [num_tokens], 0-based rank within this k-group
            ranks.append(offset[e] + within)
            # the next k-group continues after all tokens assigned to each expert so far
            offset = offset + torch.bincount(e, minlength=self.n_exp)
        return torch.stack(ranks, dim=1)  # [num_tokens, k]

    def compute_aux_loss(self, expert_probs: torch.Tensor, indices: torch.Tensor):
        """
        Computes Switch Transformer auxiliary loss (https://arxiv.org/abs/2101.03961)
        See equations (4)-(6) on page 7
        """
        with torch.no_grad():
            one_hot_indices = F.one_hot(indices, num_classes=self.n_exp)  # [B, T, k, n_exp]
            one_hot_indices = torch.sum(one_hot_indices.float(), dim=2)  # [B, T, n_exp] (sum over k dimension)
            tokens_per_expert = torch.mean(one_hot_indices.float(), dim=(0, 1))

        prob_per_expert = torch.mean(expert_probs.float(), dim=(0, 1))
        return self.n_exp * torch.sum(prob_per_expert * tokens_per_expert)

    def compute_router_z_loss(self, logits: torch.Tensor):
        """
        Computes ST-MoE router z loss (https://arxiv.org/abs/2202.08906)
        See equation (5) on page 7
        """
        z_loss = torch.logsumexp(logits, dim=-1) ** 2.0  # [B, T]
        return torch.mean(z_loss)

    def get_capacity(self, tokens_per_batch):
        capacity_factor = self.train_capacity if self.training else self.eval_capacity
        capacity = math.floor(self.top_k * capacity_factor * tokens_per_batch / self.n_exp)
        capacity += capacity % 2  # make sure capacity is an even number
        capacity = max(capacity, self.min_capacity)  # use min capacity
        assert capacity > 0
        return int(capacity)


class MLPExperts(nn.Module):
    """
    Batched MLP experts using bmm for efficiency.

    Each expert is a standard MLP: Linear(n_embd, 4*n_embd) -> GELU -> Linear(4*n_embd, n_embd).
    All experts share the same architecture but have independent weights.
    Parameters are stored as 3D tensors: [n_exp, n_embd, 4*n_embd] and [n_exp, 4*n_embd, n_embd].

    Adapted from nanoMoE MLPExperts to use nanochat's Linear class and GELU activation.
    """

    def __init__(self, config):
        super().__init__()
        self.n_exp = config.n_exp
        self.top_k = config.top_k
        self.n_embd = config.n_embd

        # Expert weights: [n_exp, n_embd, 4*n_embd] and [n_exp, 4*n_embd, n_embd]
        self.c_fc = nn.Parameter(torch.empty(config.n_exp, config.n_embd, 4 * config.n_embd))
        self.c_proj = nn.Parameter(torch.empty(config.n_exp, 4 * config.n_embd, config.n_embd))

    def forward(self, x):
        # x: [n_exp, exp_capacity, n_embd]
        # Cast expert weights to input dtype (same as nanochat's Linear class)
        c_fc = self.c_fc.to(dtype=x.dtype)
        c_proj = self.c_proj.to(dtype=x.dtype)
        x = torch.bmm(x, c_fc)  # [n_exp, exp_capacity, 4*n_embd]
        x = F.gelu(x)  # GELU activation (nanoMoE reference; bounded, unlike relu²)
        x = torch.bmm(x, c_proj)  # [n_exp, exp_capacity, n_embd]
        return x


class MOELayer(nn.Module):
    """
    Mixture of Experts layer: router + batched MLP experts.

    Replaces the standard MLP in every `stride`-th transformer block.
    All other nanochat features (RoPE, QK norm, FA3, etc.) are preserved.
    """

    def __init__(self, config, moe_manager):
        super().__init__()
        self.n_exp = config.n_exp
        self.top_k = config.top_k
        self.router = Router(config)
        self.experts = MLPExperts(config)
        self.moe_manager = moe_manager

    def forward(self, x: torch.Tensor):
        B, T, n_embd = x.size()  # track original shape of input
        num_tokens = B * T

        # pass each token through the router (sparse, O(num_tokens * top_k))
        top_k_indices, router_probs, exp_rank, exp_capacity = self.router(x)
        #   top_k_indices: [num_tokens, top_k] long  — assigned experts
        #   router_probs : [num_tokens, top_k] float — top-k softmax weights
        #   exp_rank     : [num_tokens, top_k] long  — rank within assigned expert
        #   exp_capacity : int                        — capacity per expert (drop beyond)

        # store auxiliary losses on the manager for later aggregation
        if self.router._pending_aux_loss is not None:
            self.moe_manager.add_aux_loss(self.router._pending_aux_loss)
        if self.router._pending_z_loss is not None:
            self.moe_manager.add_router_z_loss(self.router._pending_z_loss)
        if self.router._pending_expert_load is not None:
            self.moe_manager.add_expert_load(self.router._pending_expert_load)

        # flatten out the input
        x = x.view(num_tokens, n_embd)  # [num_tokens, n_embd]
        device = x.device

        # a token's k-th choice is kept only if its rank within the expert is below capacity
        valid = exp_rank < exp_capacity  # [num_tokens, top_k] bool
        # flat slot id for each (expert, rank) pair: expert-major layout of exp_batches
        slot = top_k_indices * exp_capacity + exp_rank  # [num_tokens, top_k]
        token_id = torch.arange(num_tokens, dtype=torch.long, device=device)

        # --- sparse dispatch: gather each kept token into its expert's slot ---
        # gather_idx[slot] = source token id. Empty slots (and dropped tokens) point at
        # token 0 and are never scattered back, so their garbage values don't matter.
        gather_idx = torch.zeros(self.n_exp * exp_capacity, dtype=torch.long, device=device)
        for i in range(self.top_k):
            v = valid[:, i]  # [num_tokens] bool
            gather_idx.scatter_(0, slot[v, i], token_id[v])

        exp_batches = x[gather_idx].view(self.n_exp, exp_capacity, n_embd)  # [n_exp, capacity, n_embd]
        exp_out = self.experts(exp_batches).view(-1, n_embd)  # [n_exp * capacity, n_embd]

        # --- sparse combine: scatter-add each kept expert output back to its token ---
        # cast router weights to the activation dtype so the output stays in x.dtype
        # (the dense version left it fp32, silently upcasting the residual stream).
        weights = router_probs.to(dtype=x.dtype)  # [num_tokens, top_k]
        output = torch.zeros(num_tokens, n_embd, dtype=x.dtype, device=device)
        for i in range(self.top_k):
            v = valid[:, i]  # [num_tokens] bool
            src = exp_out[slot[v, i]] * weights[v, i, None]  # [num_valid, n_embd]
            output.index_add_(0, token_id[v], src)

        # resize output before return
        return output.view(B, T, n_embd)


def init_moe_weights(module, config, n_layer):
    """
    Initialize MoE expert and router weights. Called from GPT.init_weights().
    - MLPExperts: same initialization scheme as nanochat's MLP:
        c_fc: uniform with bound = sqrt(3) * std, std = 0.4 * 1/sqrt(n_embd)
        c_proj: zeros
    - Router: Switch Transformer-style gate init (matching nanoMoE):
        w_g: trunc_normal, std = sqrt(scale / fan_in), scale = 1.0, fan_in = n_embd.
        Small std => initial logits are small => softmax is roughly uniform across
        experts, so routing starts balanced instead of collapsing to experts 0/1.
        w_noise: zeros (noise is added as softplus(w_noise(x)) * randn, so zero
        weight means no noise at init).
    """
    n_embd = config.n_embd
    s = 3**0.5 * n_embd**-0.5  # sqrt(3) multiplier

    if isinstance(module, MLPExperts):
        # Expert c_fc: uniform init with 0.4x scale (same as nanochat MLP)
        for i in range(module.n_exp):
            torch.nn.init.uniform_(module.c_fc.data[i], -s * 0.4, s * 0.4)
            torch.nn.init.zeros_(module.c_proj.data[i])
    elif isinstance(module, Router):
        # Router gate: Switch Transformer init (page 10 of https://arxiv.org/abs/2101.03961).
        # Without this the gate is left as to_empty garbage (zeros), so topk always
        # returns the lowest expert indices and routing collapses from step 0.
        w_std = (1.0 / n_embd) ** 0.5  # sqrt(scale / fan_in), scale = 1.0
        torch.nn.init.trunc_normal_(module.w_g.weight, mean=0.0, std=w_std, a=-2 * w_std, b=2 * w_std)
        if module.w_noise is not None:
            torch.nn.init.zeros_(module.w_noise.weight)
