"""ELECTRA RTD objective and original AdamW update without bias correction."""
import random
from pathlib import Path
import torch
from torch import nn
from transformers import ElectraConfig, ElectraForMaskedLM
from .model import Encoder

class AdamWNoBias(torch.optim.Optimizer):
    def __init__(self, params, lr=5e-4, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.01):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise ValueError("Sparse gradients are unsupported")
                state = self.state[p]
                if not state:
                    state.update(m=torch.zeros_like(p), v=torch.zeros_like(p))
                m, v = state["m"], state["v"]
                beta1, beta2 = group["betas"]
                m.mul_(beta1).add_(p.grad, alpha=1-beta1)
                v.mul_(beta2).addcmul_(p.grad, p.grad, value=1-beta2)
                update = m / (v.sqrt() + group["eps"])
                update.add_(p, alpha=group["weight_decay"])
                p.add_(update, alpha=-group["lr"])
        return loss

def initialize(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.trunc_normal_(module.weight, mean=0, std=0.02, a=-0.04, b=0.04)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

class RTD(nn.Module):
    def __init__(self, config, tokenizer):
        super().__init__()
        if config.hidden_size % 4 or config.intermediate_size % 4:
            raise ValueError("Generator uses exactly one quarter of hidden and feed-forward width")
        self.encoder = Encoder(config)
        self.discriminator_head = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size), nn.GELU(), nn.Linear(config.hidden_size, 1))
        gc = ElectraConfig(vocab_size=config.vocab_size, embedding_size=config.embedding_size,
                           hidden_size=config.hidden_size//4, intermediate_size=config.intermediate_size//4,
                           num_hidden_layers=config.num_layers, num_attention_heads=1,
                           max_position_embeddings=config.max_position_embeddings, pad_token_id=config.pad_token_id,
                           hidden_dropout_prob=config.dropout, attention_probs_dropout_prob=config.dropout)
        self.generator = ElectraForMaskedLM(gc)
        self.generator.apply(initialize)
        self.encoder.apply(initialize)
        self.discriminator_head.apply(initialize)
        self.encoder.embeddings.word_embeddings.weight = self.generator.electra.embeddings.word_embeddings.weight
        with torch.no_grad():
            self.encoder.embeddings.word_embeddings.weight[config.pad_token_id].zero_()
        self.special_ids = tuple(tokenizer.all_special_ids)
        self.mask_id = tokenizer.mask_token_id

    def forward(self, ids, attention_mask):
        eligible = attention_mask.bool().clone()
        for special in self.special_ids:
            eligible &= ids != special
        selected = torch.bernoulli(eligible.float() * 0.15).bool()
        if not selected.any():
            if not eligible.any():
                raise ValueError("Pretraining batch contains no maskable tokens")
            selected.view(-1)[eligible.view(-1).nonzero()[0, 0]] = True
        masked = ids.clone()
        masked[selected & (torch.rand(ids.shape, device=ids.device) < 0.85)] = self.mask_id
        labels = ids.masked_fill(~selected, -100)
        output = self.generator(masked, attention_mask=attention_mask, labels=labels)
        with torch.no_grad():
            logits = output.logits.float()
            noise = torch.rand_like(logits).clamp_(1e-6, 1-1e-6)
            sampled = (logits - torch.log(-torch.log(noise))).argmax(-1)
            replaced = ids.clone()
            replaced[selected] = sampled[selected]
            targets = (replaced != ids).float()
        predictions = self.discriminator_head(self.encoder(replaced, attention_mask)).squeeze(-1)
        disc_loss = torch.nn.functional.binary_cross_entropy_with_logits(predictions.float()[attention_mask.bool()], targets[attention_mask.bool()])
        return output.loss + 50 * disc_loss, output.loss.detach(), disc_loss.detach()

class CorpusSampler:
    """Bounded-memory sampling from corpus bytes, deterministic in one process."""
    def __init__(self, directory, batch_size, seed):
        self.files = sorted(p for p in Path(directory).glob("*.utf8") if p.stat().st_size)
        if not self.files:
            raise ValueError("No nonempty .utf8 pretraining files")
        self.sizes = [p.stat().st_size for p in self.files]
        self.batch_size = batch_size
        self.rng = random.Random(seed)
        self.pending = []
    def batch(self):
        attempts = 0
        while len(self.pending) < self.batch_size * 4:
            attempts += 1
            if attempts > 1000:
                raise ValueError("Corpus must contain enough segments of at least 50 whitespace words")
            path = self.rng.choices(self.files, weights=self.sizes)[0]
            size = path.stat().st_size
            width = max(65536, self.batch_size * 4 * 150 * 12)
            with path.open("rb") as f:
                if size <= width:
                    raw = f.read()
                else:
                    f.seek(self.rng.randrange(size))
                    raw = f.read(width)
                    if len(raw) < width:
                        f.seek(0)
                        raw += f.read(width-len(raw))
            words = raw.decode("utf-8", errors="ignore").split()
            if size > width:
                words = words[1:-1]
            segments = [" ".join(words[i:i+150]) for i in range(0, len(words), 150) if len(words[i:i+150]) >= 50]
            self.rng.shuffle(segments)
            self.pending.extend(segments[:self.batch_size*4])
            self.rng.shuffle(self.pending)
        result, self.pending = self.pending[:self.batch_size], self.pending[self.batch_size:]
        return result
    def state_dict(self):
        return {"rng": self.rng.getstate(), "pending": self.pending}
    def load_state_dict(self, state):
        self.rng.setstate(state["rng"])
        self.pending = state["pending"]
