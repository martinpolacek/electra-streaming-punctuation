from dataclasses import asdict, dataclass

FAMILIES = {"Small": (128, 256, 1024, 4), "Mini": (128, 128, 512, 2), "Tiny": (64, 96, 384, 2)}
PAPER_VARIANTS = {"Small": (12, 6, 4, 3), "Mini": (12, 6), "Tiny": (4,)}
LABELS = ["NONE", "QUESTION", "PERIOD", "COMMA"]
CLASS_WEIGHTS = [1.0, 5.0, 2.5, 1.5]
TOKENIZER_ID = "AILabTUL/mELECTRA"
TOKENIZER_REVISION = "b4f2d443cff9c9ca2c9a63c246c6c343a9b5396d"

@dataclass(frozen=True)
class Config:
    embedding_size: int = 128
    hidden_size: int = 256
    intermediate_size: int = 1024
    num_heads: int = 4
    num_layers: int = 6
    vocab_size: int = 30523
    pad_token_id: int = 0
    max_position_embeddings: int = 512
    dropout: float = 0.1

    def __post_init__(self):
        for key in ("embedding_size", "hidden_size", "intermediate_size", "num_heads", "num_layers", "vocab_size", "max_position_embeddings"):
            if not isinstance(getattr(self, key), int) or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if not 0 <= self.pad_token_id < self.vocab_size or not 0 <= self.dropout < 1:
            raise ValueError("Invalid padding ID or dropout")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def family(cls, name="Small", layers=6, **kwargs):
        e, h, f, a = FAMILIES[name]
        return cls(embedding_size=e, hidden_size=h, intermediate_size=f,
                   num_heads=a, num_layers=layers, **kwargs)
