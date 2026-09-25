import math
import torch
import torch.nn.functional as F


class OutroController:
    def __init__(
        self,
        num_layers: int,
        start_layer: int = 0,
        end_layer: int = None,
        apply_once: bool = True,
        med_mul: float = 1000.0,
        abs_thr: float = 100.0,
        num_heads: int = 1,
        num_kv_heads: int = None,
        head_dim: int = 1,
        gamma: float = 3.0,
        temp: float = 0.1,
        rescale: bool = True,
        enable_relu_tanh: bool = True,
    ):
        self.num_layers = int(num_layers)
        self.start_layer = int(start_layer)
        self.end_layer = int(end_layer) if end_layer is not None else int(num_layers)
        self.apply_once = bool(apply_once)
        self.apply_interval = max(1, int(num_layers) // 7)
        self.med_mul = float(med_mul)
        self.abs_thr = float(abs_thr)
        self.num_heads = int(num_heads)
        self.num_kv_heads = int(num_kv_heads) if num_kv_heads is not None else self.num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = int(head_dim)
        self.gamma = float(gamma)
        self.temp = float(temp)
        self.rescale = bool(rescale)
        self.enable_relu_tanh = bool(enable_relu_tanh)

        self.sink_indices_cache = {}
        self.applied_sinks = set()
        self.cumulative_sinks = set()
        self.v_sink_proj_cache = {}
        self.delta_v_cache = {}

    def _is_enhancement_layer(self, layer_idx: int) -> bool:
        return layer_idx % self.apply_interval == 0

    @torch.no_grad()
    def reset_runtime(self):
        self.sink_indices_cache = {}
        self.applied_sinks = set()
        self.cumulative_sinks = set()
        self.v_sink_proj_cache = {}
        self.delta_v_cache = {}

    @torch.no_grad()
    def cache_sink_indices(self, layer_idx: int, hidden_states: torch.Tensor):
        if layer_idx < self.start_layer or layer_idx >= self.end_layer:
            return
        if hidden_states is None or hidden_states.dim() != 3:
            return
        if hidden_states.size(1) <= 1:
            return
        h0 = hidden_states[0]
        abs_h = h0.abs()
        global_median = float(abs_h.median().item())
        threshold = max(self.abs_thr, self.med_mul * global_median)
        max_abs_per_token = abs_h.max(dim=-1).values
        sink_mask = max_abs_per_token >= threshold
        sink_indices = torch.nonzero(sink_mask, as_tuple=False).squeeze(-1).tolist()
        self.sink_indices_cache[layer_idx] = sink_indices

    @torch.no_grad()
    def cache_v_proj_sink(self, layer_idx: int, value_states: torch.Tensor):
        if layer_idx < self.start_layer or layer_idx >= self.end_layer:
            return
        sink_indices = self.sink_indices_cache.get(layer_idx, [])
        if not sink_indices:
            return
        S = value_states.shape[2]
        if S <= 1:
            return
        valid = [i for i in sink_indices if i < S]
        if not valid:
            return
        sink_idx = torch.as_tensor(valid, dtype=torch.long, device=value_states.device)
        v_sink = value_states[0, :, sink_idx, :].mean(dim=1)
        self.v_sink_proj_cache[layer_idx] = v_sink.detach()

    @torch.no_grad()
    def compute_delta_v(self, layer_idx: int, query_states, key_states, value_states):
        if not self._is_enhancement_layer(layer_idx):
            return
        sink_indices = self.sink_indices_cache.get(layer_idx, [])
        if not sink_indices:
            return
        B, _, S, D = query_states.shape
        if S <= 1:
            return
        valid = [i for i in sink_indices if i < S]
        if not valid:
            return
        if self.apply_once:
            valid = [i for i in valid if i not in self.applied_sinks]
            if not valid:
                return

        sink_idx = torch.as_tensor(valid, dtype=torch.long, device=query_states.device)
        q_sink = query_states[:, :, sink_idx, :]

        k_exp = key_states.repeat_interleave(self.num_kv_groups, dim=1)
        v_exp = value_states.repeat_interleave(self.num_kv_groups, dim=1)

        attn_w = torch.matmul(q_sink, k_exp.transpose(-2, -1)) / math.sqrt(D)
        attn_p = F.softmax(attn_w, dim=-1)
        delta_v = torch.matmul(attn_p, v_exp)

        self.delta_v_cache[layer_idx] = (delta_v.transpose(1, 2).reshape(B, len(valid), -1), valid)

    def get_sink_indices(self, layer_idx: int, seq_len: int):
        indices = self.sink_indices_cache.get(layer_idx, [])
        if not indices:
            return []
        valid = [i for i in indices if 0 <= i < seq_len]
        if self.apply_once:
            valid = [i for i in valid if i not in self.applied_sinks]
        return valid

    @torch.no_grad()
    def mark_applied(self, sink_indices):
        if not self.apply_once:
            return
        for idx in sink_indices:
            self.applied_sinks.add(int(idx))

    @torch.no_grad()
    def apply_relu_tanh_gating(
        self, layer_idx, hidden_states, current_sinks, seq_len, bsz,
    ) -> torch.Tensor:
        if not (self.num_heads > 0 and self.head_dim > 0):
            return hidden_states

        non_sink_mask = torch.ones(seq_len, dtype=torch.bool, device=hidden_states.device)
        attn = hidden_states.view(bsz, seq_len, self.num_heads, self.head_dim)

        if current_sinks:
            cumulative = [i for i in self.cumulative_sinks if 0 <= i < seq_len]
            if cumulative:
                non_sink_mask[torch.as_tensor(cumulative, dtype=torch.long, device=hidden_states.device)] = False

        v_sink_kv = self.v_sink_proj_cache.get(layer_idx)
        if v_sink_kv is None:
            return hidden_states
        if v_sink_kv.device != hidden_states.device or v_sink_kv.dtype != hidden_states.dtype:
            v_sink_kv = v_sink_kv.to(device=hidden_states.device, dtype=hidden_states.dtype)
            self.v_sink_proj_cache[layer_idx] = v_sink_kv

        if not non_sink_mask.any():
            return hidden_states

        target = attn[:, non_sink_mask, :, :]
        S_ns = target.shape[1]
        if S_ns == 0:
            return hidden_states

        v_sink = v_sink_kv.repeat_interleave(self.num_kv_groups, dim=0)

        orig_norm = target.norm(dim=-1, keepdim=True)

        v_n = F.normalize(v_sink, dim=-1)
        o_n = F.normalize(target, dim=-1)
        cos = (v_n * o_n).sum(dim=-1, keepdim=True)
        mask = torch.tanh(F.relu(cos) / self.temp)

        v_norm_sq = (v_sink ** 2).sum(dim=-1, keepdim=True)
        dot = (target * v_sink).sum(dim=-1, keepdim=True).abs()
        proj = (dot / (v_norm_sq + 1e-8)) * v_sink
        mod = target + mask * (self.gamma * proj)

        if self.rescale:
            new_norm = mod.norm(dim=-1, keepdim=True)
            mod = mod * (orig_norm / (new_norm + 1e-8))

        result = attn.clone()
        result[:, non_sink_mask, :, :] = mod
        return result.view(bsz, seq_len, -1)

    @torch.no_grad()
    def apply_sink_enhancement(
        self, layer_idx, hidden_states, sink_indices, seq_len, bsz,
    ) -> torch.Tensor:
        if not sink_indices:
            return hidden_states

        sink_idx = torch.as_tensor(sink_indices, dtype=torch.long, device=hidden_states.device)
        out = hidden_states.clone()

        cached = self.delta_v_cache.get(layer_idx)
        if cached is not None and cached[1] == sink_indices:
            replacement = cached[0]
        else:
            non_sink_mask = torch.ones(seq_len, dtype=torch.bool, device=hidden_states.device)
            non_sink_mask[sink_idx] = False
            if non_sink_mask.any():
                base = hidden_states[:, non_sink_mask, :].mean(dim=1, keepdim=True)
            else:
                base = hidden_states.mean(dim=1, keepdim=True)
            replacement = base.expand(bsz, sink_idx.numel(), -1)

        out.index_copy_(1, sink_idx, replacement)
        self.mark_applied(sink_indices)
        return out

    @torch.no_grad()
    def apply_outro(self, layer_idx: int, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states is None or hidden_states.dim() != 3:
            return hidden_states

        bsz, seq_len, _ = hidden_states.shape
        current_sinks_raw = self.sink_indices_cache.get(layer_idx, [])
        current_sinks = [i for i in current_sinks_raw if 0 <= i < seq_len]
        for idx in current_sinks:
            self.cumulative_sinks.add(int(idx))

        sink_indices = self.get_sink_indices(layer_idx, seq_len)
        do_update = bool(sink_indices) and self._is_enhancement_layer(layer_idx)
        do_gating = self.enable_relu_tanh and (layer_idx in self.v_sink_proj_cache)
        if not do_update and not do_gating:
            return hidden_states

        out = hidden_states
        if do_update:
            out = self.apply_sink_enhancement(
                layer_idx=layer_idx,
                hidden_states=out,
                sink_indices=sink_indices,
                seq_len=seq_len,
                bsz=bsz,
            )

        if do_gating:
            out = self.apply_relu_tanh_gating(
                layer_idx=layer_idx,
                hidden_states=out,
                current_sinks=current_sinks,
                seq_len=seq_len,
                bsz=bsz,
            )

        return out
