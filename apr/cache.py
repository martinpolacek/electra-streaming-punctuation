"""Approximate bidirectional KV reuse; full repair never mutates the cache."""
import torch
from torch import nn

class KVCache:
    def __init__(self, model):
        if model.training:
            raise ValueError("Cache requires model.eval()")
        if model.config.max_position_embeddings < 128:
            raise ValueError("Cache requires at least 128 position embeddings")
        self.model = model
        self.reset()

    def reset(self):
        self.origin = None
        self.bounds = None
        self.cache = None
        self.last_stats = {}

    @torch.inference_mode()
    def step(self, ids, a, b, word_start, target, ends):
        if self.model.training:
            raise ValueError("Cache requires model.eval()")
        if ids.shape != (1, b-a) or not a <= word_start <= target < b or b-a > 64:
            raise ValueError("Invalid 64-subword cache window")
        if ends.ndim != 1 or not 1 <= len(ends) <= 5 or int(ends[0]) != target or int(ends[-1]) != b-1 or (ends[1:] <= ends[:-1]).any():
            raise ValueError("Expected current and up to four future word ends")
        previous_end = b if self.bounds is None else self.bounds[1]
        if self.bounds is not None:
            pa, pb = self.bounds
            if a < pa or b < pb or a > pb or word_start > pb:
                raise ValueError("Discontinuous stream; reset cache before each recording")
        refresh = self.origin is None or b - self.origin > 128
        if refresh:
            self.origin = a
        active = a if refresh else word_start
        prefix = active-a
        back = self.model.backbone
        positions = torch.arange(active-self.origin, b-self.origin, device=ids.device)[None, :]
        h = back.embed(ids[:, prefix:], positions)
        next_cache = []
        for index, layer in enumerate(back.encoder.layer):
            att = layer.attention.self
            q, k, v = [att.split(proj(h)) for proj in (att.query, att.key, att.value)]
            if prefix:
                pa, _ = self.bounds
                oldk, oldv = self.cache[index]
                k = torch.cat((oldk[:, :, a-pa:active-pa], k), dim=2)
                v = torch.cat((oldv[:, :, a-pa:active-pa], v), dim=2)
            next_cache.append((k, v))
            weights = ((q @ k.transpose(-2, -1)) * att.scale).softmax(-1)
            context = (weights @ v).transpose(1, 2).contiguous().view_as(h)
            h = layer.ffn(layer.attention.output(context, h))
        selected = h[0, target-active]
        logits = self.model.final_head(selected)
        metadata = logits.new_tensor([(len(ends)-1)/4, (target-word_start+1)/8, (target-a)/63,
                                      (target-self.origin)/127, prefix/64, (b-a)/64,
                                      float(active == a), (b-previous_end)/8])
        features = torch.cat((logits.softmax(-1), selected, metadata))
        self.cache, self.bounds = next_cache, (a, b)
        self.last_stats = {"origin": self.origin, "active_tokens": b-active, "full_refresh": active == a}
        return logits, features

class Gate(nn.Module):
    def __init__(self, dimension=268):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dimension, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)

def confidence_score(features, kind):
    p = features[..., :4]
    if kind == "margin":
        top = p.topk(2, dim=-1).values
        return 1 - (top[..., 0] - top[..., 1])
    if kind == "entropy":
        return -(p * p.clamp_min(1e-9).log()).sum(-1)
    raise ValueError(kind)
