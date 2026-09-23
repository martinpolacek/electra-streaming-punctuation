"""Dense ELECTRA with the projection used by all seven archived architectures.

Even Mini (embedding width == hidden width) retains its learned projection.
A stock HF ElectraModel omits that projection for equal widths.
"""
import torch
from torch import nn
from .config import Config

class Embeddings(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.word_embeddings = nn.Embedding(c.vocab_size, c.embedding_size, padding_idx=c.pad_token_id)
        self.position_embeddings = nn.Embedding(c.max_position_embeddings, c.embedding_size)
        self.token_type_embeddings = nn.Embedding(2, c.embedding_size)
        self.LayerNorm = nn.LayerNorm(c.embedding_size, eps=1e-12)
        self.dropout = nn.Dropout(c.dropout)

    def forward(self, ids, positions=None, token_types=None):
        if positions is None:
            positions = torch.arange(ids.shape[1], device=ids.device)[None, :]
        if token_types is None:
            token_types = torch.zeros_like(ids)
        h = self.word_embeddings(ids) + self.position_embeddings(positions) + self.token_type_embeddings(token_types)
        return self.dropout(self.LayerNorm(h))

class SelfAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.num_heads, self.head_dim = c.num_heads, c.hidden_size // c.num_heads
        self.scale = self.head_dim ** -0.5
        self.query = nn.Linear(c.hidden_size, c.hidden_size)
        self.key = nn.Linear(c.hidden_size, c.hidden_size)
        self.value = nn.Linear(c.hidden_size, c.hidden_size)
        self.dropout = nn.Dropout(c.dropout)

    def split(self, x):
        return x.view(*x.shape[:2], self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, h, mask):
        q, k, v = [self.split(proj(h)) for proj in (self.query, self.key, self.value)]
        scores = (q @ k.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(~mask[:, None, None, :].bool(), -1e4)
        return (self.dropout(scores.softmax(-1)) @ v).transpose(1, 2).contiguous().view_as(h)

class AttentionOutput(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.dense = nn.Linear(c.hidden_size, c.hidden_size)
        self.dropout = nn.Dropout(c.dropout)
        self.LayerNorm = nn.LayerNorm(c.hidden_size, eps=1e-12)
    def forward(self, h, residual):
        return self.LayerNorm(self.dropout(self.dense(h)) + residual)

class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.self = SelfAttention(c)
        self.output = AttentionOutput(c)
    def forward(self, h, mask):
        return self.output(self.self(h, mask), h)

class FFN(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.dense_up = nn.Linear(c.hidden_size, c.intermediate_size)
        self.dense_down = nn.Linear(c.intermediate_size, c.hidden_size)
        self.dropout = nn.Dropout(c.dropout)
        self.LayerNorm = nn.LayerNorm(c.hidden_size, eps=1e-12)
    def forward(self, h):
        return self.LayerNorm(self.dropout(self.dense_down(torch.nn.functional.gelu(self.dense_up(h)))) + h)

class Layer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.attention, self.ffn = Attention(c), FFN(c)
    def forward(self, h, mask):
        return self.ffn(self.attention(h, mask))

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = Embeddings(config)
        self.embeddings_project = nn.Linear(config.embedding_size, config.hidden_size)
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList([Layer(config) for _ in range(config.num_layers)])
    def embed(self, ids, positions=None):
        return self.embeddings_project(self.embeddings(ids, positions))
    def forward(self, ids, mask, positions=None):
        h = self.embed(ids, positions)
        for layer in self.encoder.layer:
            h = layer(h, mask)
        return h

def head(width):
    return nn.Sequential(nn.Linear(width, 512), nn.SELU(), nn.Linear(512, 4))

class Punctuator(nn.Module):
    def __init__(self, config, exits=None):
        super().__init__()
        self.config = config
        self.exits = tuple(exits if exits is not None else [config.num_layers])
        if not self.exits or self.exits != tuple(sorted(set(self.exits))) or self.exits[-1] != config.num_layers or self.exits[0] < 1:
            raise ValueError("Exits must be distinct increasing depths ending at the final layer")
        self.backbone = Encoder(config)
        self.heads = nn.ModuleDict({str(d): head(config.hidden_size) for d in self.exits})
    @property
    def final_head(self):
        return self.heads[str(self.config.num_layers)]
    def forward(self, ids, mask):
        h = self.backbone.embed(ids)
        outputs = []
        for depth, layer in enumerate(self.backbone.encoder.layer, 1):
            h = layer(h, mask)
            if depth in self.exits:
                outputs.append(self.heads[str(depth)](h))
        return outputs
    def add_exits(self, depths=None):
        depths = tuple(depths if depths is not None else range(2, self.config.num_layers + 1))
        if len(depths) < 2 or tuple(sorted(set(depths))) != depths or depths[0] < 2 or depths[-1] != self.config.num_layers:
            raise ValueError("Exit training needs increasing depths from layer 2 through the final layer")
        for depth in depths:
            if str(depth) not in self.heads:
                self.heads[str(depth)] = head(self.config.hidden_size).to(next(self.parameters()).device)
        self.exits = depths
        for p in self.parameters():
            p.requires_grad_(False)
        for depth in depths[:-1]:
            for p in self.heads[str(depth)].parameters():
                p.requires_grad_(True)

@torch.inference_mode()
def target_logits(model, ids, mask, targets, threshold=None, all_exits=False):
    if model.training:
        raise ValueError("Call model.eval() before inference")
    if ids.ndim != 2 or mask.shape != ids.shape or targets.shape != (ids.shape[0],):
        raise ValueError("Expected ids/mask [batch,tokens], targets [batch]")
    if ids.dtype != torch.long or targets.dtype != torch.long or mask.dtype != torch.bool:
        raise ValueError("Expected int64 IDs/targets and boolean mask")
    if ids.numel() == 0 or ((targets < 0) | (targets >= ids.shape[1])).any():
        raise ValueError("Invalid target position")
    rows = torch.arange(len(ids), device=ids.device)
    if not mask[rows, targets].all():
        raise ValueError("Target is padding")
    if threshold is not None and not 0 <= threshold <= 1:
        raise ValueError("Exit threshold must be between zero and one")
    if threshold is not None and len(model.exits) < 2:
        raise ValueError("Early exit needs trained intermediate heads")
    h = model.backbone.embed(ids)
    active = rows
    result = h.new_empty((len(ids), 4))
    depths = targets.new_empty(len(ids))
    every = []
    for depth, layer in enumerate(model.backbone.encoder.layer, 1):
        h = layer(h, mask)
        if depth not in model.exits or (threshold is None and not all_exits and depth != model.config.num_layers):
            continue
        logits = model.heads[str(depth)](h[torch.arange(len(h), device=h.device), targets])
        if all_exits:
            every.append(logits)
            continue
        done = torch.ones(len(h), dtype=torch.bool, device=h.device) if depth == model.config.num_layers else logits.float().softmax(-1).amax(-1) >= threshold
        result[active[done]], depths[active[done]] = logits[done], depth
        remaining = (~done).nonzero(as_tuple=False).flatten()
        if not len(remaining):
            break
        active, h, mask, targets = [x.index_select(0, remaining) for x in (active, h, mask, targets)]
    return torch.stack(every, dim=1) if all_exits else (result, depths)
