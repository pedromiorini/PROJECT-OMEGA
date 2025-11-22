# src/omega/core/model.py
# =============================================================================
# OMEGA-CORE-V1-1.4B — Modelo de Linguagem Soberano
# Arquitetura: Híbrida Mamba-2 + MoE com Segurança Embutida
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba2
from torch.utils.checkpoint import checkpoint
import math
from typing import Optional

# --- CONFIGURAÇÃO DO MODELO ---
class OmegaCoreConfig:
    d_model = 2048
    n_layer = 36
    vocab_size = 32000
    d_state = 64
    expand = 2
    d_conv = 4
    num_experts = 8
    top_k = 2
    context_length = 32768
    moe_every = 2
    use_bias = False
    rope_theta = 10000.0
    use_gradient_checkpointing = True
    load_balancing_loss_coef = 0.01
    safety_threshold = 0.7

config = OmegaCoreConfig()

# --- COMPONENTES DO MODELO ---

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight

class SafetyGuard(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.safety_proj = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.SiLU(), nn.Linear(d_model // 2, 128))
        self.harm_head = nn.Linear(128, 5)
        self.uncertainty_head = nn.Linear(128, 1)
        self.toxicity_head = nn.Linear(128, 1)
        
    def forward(self, hidden):
        safety_features = self.safety_proj(hidden)
        harm_logits = self.harm_head(safety_features)
        uncertainty = torch.sigmoid(self.uncertainty_head(safety_features))
        toxicity = torch.sigmoid(self.toxicity_head(safety_features))
        risk_score = torch.sigmoid(harm_logits).max(dim=-1)
        return risk_score, uncertainty.squeeze(-1), toxicity.squeeze(-1), harm_logits

class Expert(nn.Module):
    def __init__(self, d_model, expansion=4):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_model * expansion, bias=False)
        self.w2 = nn.Linear(d_model * expansion, d_model, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.w2(self.act(self.w1(x)))

class SparseMoE(nn.Module):
    def __init__(self, d_model, num_experts, top_k):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([Expert(d_model) for _ in range(num_experts)])

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)
        gate_logits = self.gate(x_flat)
        weights, selected_experts = torch.topk(gate_logits, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)
        output = torch.zeros_like(x_flat)
        expert_mask = F.one_hot(selected_experts, self.num_experts).float()
        
        for i, expert in enumerate(self.experts):
            token_mask = expert_mask[..., i].sum(dim=-1) > 0
            if token_mask.any():
                expert_input = x_flat[token_mask]
                expert_output = expert(expert_input)
                expert_weights = (expert_mask[..., i] * weights).sum(dim=-1, keepdim=True)[token_mask]
                output[token_mask] += expert_output * expert_weights
        
        gate_probs = F.softmax(gate_logits, dim=-1)
        load_balancing_loss = self.num_experts * (gate_probs.mean(0) ** 2).sum()
        return output.view(batch_size, seq_len, d_model), load_balancing_loss

class OmegaMambaBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.mamba = Mamba2(d_model=config.d_model, d_state=config.d_state, d_conv=config.d_conv, expand=config.expand, headdim=64, use_bias=config.use_bias)
        self.norm_mamba = RMSNorm(config.d_model)
        self.use_moe = (layer_idx % config.moe_every == 0)
        if self.use_moe:
            self.moe = SparseMoE(config.d_model, config.num_experts, config.top_k)
            self.norm_moe = RMSNorm(config.d_model)
        self.safety = SafetyGuard(config.d_model)

    def forward(self, x):
        residual = x
        x = self.norm_mamba(x)
        x = self.mamba(x) + residual
        load_balancing_loss = 0.0
        if self.use_moe:
            x_moe, lb_loss = self.moe(self.norm_moe(x))
            x = x + x_moe
            load_balancing_loss = lb_loss
        risk_score, uncertainty, toxicity, harm_logits = self.safety(x)
        return x, risk_score, uncertainty, toxicity, harm_logits, load_balancing_loss

class OmegaCoreModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([OmegaMambaBlock(config, i) for i in range(config.n_layer)])
        self.norm_f = RMSNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.head.weight = self.embed.weight
        self.initialize_weights()

    def initialize_weights(self):
        for name, p in self.named_parameters():
            if 'embed' in name or ('head' in name and p.dim() > 1):
                nn.init.normal_(p, mean=0.0, std=0.02)
            elif p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=1.0 / math.sqrt(self.config.n_layer))

    def _forward_block(self, block, x):
        return block(x)

    def forward(self, input_ids, labels=None, safety_labels=None):
        x = self.embed(input_ids)
        all_risk_scores, all_uncertainties, all_toxicities, all_harm_logits = [], [], [], []
        total_lb_loss = 0.0
        
        for block in self.blocks:
            if self.config.use_gradient_checkpointing and self.training:
                x, risk, unc, tox, harm, lb = checkpoint(self._forward_block, block, x)
            else:
                x, risk, unc, tox, harm, lb = block(x)
            all_risk_scores.append(risk); all_uncertainties.append(unc); all_toxicities.append(tox); all_harm_logits.append(harm); total_lb_loss += lb
        
        x = self.norm_f(x)
        logits = self.head(x)
        
        weights = torch.softmax(torch.linspace(0, 1, len(all_risk_scores), device=x.device), dim=0)
        final_risk = sum(w * r for w, r in zip(weights, all_risk_scores))
        final_uncertainty = sum(w * u for w, u in zip(weights, all_uncertainties))
        final_toxicity = sum(w * t for w, t in zip(weights, all_toxicities))
        
        loss, safety_metrics = None, {'risk_score': final_risk.mean().item(), 'uncertainty': final_uncertainty.mean().item(), 'toxicity': final_toxicity.mean().item()}
        
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            lm_loss = F.cross_entropy(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1), ignore_index=-100)
            safety_loss = 0.0
            if safety_labels is not None:
                safety_loss = F.binary_cross_entropy(final_risk[..., :-1], safety_labels[..., 1:].float()) * 0.1
            lb_loss = total_lb_loss * self.config.load_balancing_loss_coef / len(self.blocks)
            loss = lm_loss + safety_loss + lb_loss
            safety_metrics.update({'lm_loss': lm_loss.item(), 'safety_loss': safety_loss if isinstance(safety_loss, float) else safety_loss.item(), 'lb_loss': lb_loss.item()})
        
        return logits, loss, safety_metrics

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=100, temperature=0.8, top_p=0.9, safety_check=True):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = input_ids if input_ids.size(1) <= self.config.context_length else input_ids[:, -self.config.context_length:]
            logits, _, safety_metrics = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if safety_check and safety_metrics['risk_score'] > self.config.safety_threshold:
                print(f"⚠️ Geração bloqueada (risk: {safety_metrics['risk_score']:.3f})")
                return input_ids, safety_metrics
            
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            logits[sorted_indices[sorted_indices_to_remove]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if next_token.item() == 0: break
        return input_ids, safety_metrics
