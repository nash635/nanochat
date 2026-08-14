"""
MoE (Mixture of Experts) module for nanochat.

Ports the Router, MLPExperts, and MOELayer from nanoMoE, adapted to:
- nanochat's Linear class (casts weights to input dtype in forward)
- nanochat's relu² activation (F.relu(x).square())
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

    def reset(self):
        self.aux_loss = []
        self.router_z_loss = []

    def add_aux_loss(self, loss):
        self.aux_loss.append(loss)

    def add_router_z_loss(self, loss):
        self.router_z_loss.append(loss)

    def aggregate_aux_loss(self):
        return sum(self.aux_loss)

    def aggregate_router_z_loss(self):
        return sum(self.router_z_loss)


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

            # normalize expert probabilities over top-k
            router_probs = torch.full_like(logits, float('-inf'))  # [B, T, n_exp]
            router_probs.scatter_(-1, top_k_indices, top_k_logits)
            router_probs = F.softmax(router_probs, dim=-1)

            # compute auxiliary load balancing loss
            # this loss encourages equal probability assigned to each expert
            # and equal load balancing of tokens assigned to each expert
            if self.use_aux_loss:
                aux_loss = self.compute_aux_loss(router_probs, top_k_indices)
                self._pending_aux_loss = aux_loss
            else:
                self._pending_aux_loss = None

            # compute expert capacity
            exp_capacity = self.get_capacity(num_tokens)

            # make a multi-hot mask of chosen experts, size [B, T, n_exp]
            # entries are 0 if expert not chosen and 1 if expert chosen
            exp_mask = F.one_hot(top_k_indices, num_classes=self.n_exp)  # [B, T, k, n_exp]
            exp_mask = exp_mask.view(num_tokens, self.top_k, self.n_exp)  # [B * T, k, n_exp]
            exp_mask = exp_mask.permute(1, 0, 2)  # [k, B * T, n_exp]

            # compute cumulative sum of each token over experts, this stores
            # the index of each token within the batch of each expert
            exp_rank = exp_mask.reshape(self.top_k * num_tokens, self.n_exp)  # [k * B * T, n_exp]
            exp_rank = torch.cumsum(exp_rank, dim=0) - 1  # cumulative sum of expert selections [k * B * T, n_exp]
            exp_rank = exp_rank.reshape(self.top_k, num_tokens, self.n_exp)  # [k, B * T, n_exp]

            # mask out (set to zero) entries that go beyond expert capacity
            exp_mask *= torch.lt(exp_rank, exp_capacity)  # [k, B * T, n_exp]
            used_capacity = torch.sum(exp_mask, dim=(0, 1))  # [n_exp]

            # mask rank to only include tokens that are selected
            exp_rank = torch.sum(exp_mask * exp_rank, dim=-1)  # [k, B * T]

            # mask probabilities to only include selected experts
            router_probs = router_probs.view(num_tokens, self.n_exp)[None, :]  # [1, B * T, n_exp]
            exp_weights = exp_mask * router_probs  # [k, B * T, n_exp]

            # convert rank into one-hot vectors over the available capacity
            exp_rank_sc = F.one_hot(exp_rank, num_classes=exp_capacity)  # [k, B * T, exp_capacity]

            # create a vector that stores, for each token, the weight of selected
            # experts at token's position in the capacity of that expert
            cb_weight = torch.sum(exp_weights.unsqueeze(3) * exp_rank_sc.unsqueeze(2), dim=0)  # [B * T, n_exp, exp_capacity]
            sec_mask = cb_weight.bool()  # binary mask of selected experts for each token
            return used_capacity, cb_weight, sec_mask

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

    Each expert is a standard MLP: Linear(n_embd, 4*n_embd) -> relu² -> Linear(4*n_embd, n_embd).
    All experts share the same architecture but have independent weights.
    Parameters are stored as 3D tensors: [n_exp, n_embd, 4*n_embd] and [n_exp, 4*n_embd, n_embd].

    Adapted from nanoMoE MLPExperts to use nanochat's Linear class and relu² activation.
    """

    def __init__(self, config):
        super().__init__()
        self.n_exp = config.n_exp
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
        x = F.relu(x).square()  # relu² activation (nanochat style)
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
        self.router = Router(config)
        self.experts = MLPExperts(config)
        self.moe_manager = moe_manager

    def forward(self, x: torch.Tensor):
        B, T, n_embd = x.size()  # track original shape of input
        num_tokens = B * T

        # pass each token through the router
        used_capacity, exp_weight, exp_mask = self.router(x)

        # store auxiliary losses on the manager for later aggregation
        if self.router._pending_aux_loss is not None:
            self.moe_manager.add_aux_loss(self.router._pending_aux_loss)
        if self.router._pending_z_loss is not None:
            self.moe_manager.add_router_z_loss(self.router._pending_z_loss)

        # flatten out the input
        x = x.view(num_tokens, n_embd)

        # reshape tokens into batches for each expert
        # [n_exp, exp_capacity, B * T] * [B * T, n_embd] -> [n_exp, exp_capacity, n_embd]
        exp_batches = exp_mask.permute(1, 2, 0).type_as(x) @ x

        # compute expert output
        exp_out = self.experts(exp_batches)  # [n_exp, exp_capacity, n_embd]

        # aggregate expert outputs based on router weights
        # eq (2) on page 4 of ST-MoE (https://arxiv.org/abs/2202.08906)
        exp_weight = exp_weight.view(num_tokens, -1)  # [B * T, n_exp * exp_capacity]
        exp_out = exp_out.view(-1, n_embd)  # [n_exp * exp_capacity, n_embd]
        output = exp_weight @ exp_out  # [B * T, n_embd]

        # resize output before return
        return output.view(B, T, n_embd)


def init_moe_weights(module, config, n_layer):
    """
    Initialize MoE expert weights. Called from GPT.init_weights().
    Uses the same initialization scheme as nanochat's MLP:
    - c_fc: uniform with bound = sqrt(3) * std, std = 0.4 * 1/sqrt(n_embd)
    - c_proj: zeros
    """
    n_embd = config.n_embd
    s = 3**0.5 * n_embd**-0.5  # sqrt(3) multiplier

    if isinstance(module, MLPExperts):
        # Expert c_fc: uniform init with 0.4x scale (same as nanochat MLP)
        for i in range(module.n_exp):
            torch.nn.init.uniform_(module.c_fc.data[i], -s * 0.4, s * 0.4)
            torch.nn.init.zeros_(module.c_proj.data[i])
