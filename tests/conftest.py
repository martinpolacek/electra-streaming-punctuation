import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, normalizers
from transformers import PreTrainedTokenizerFast
from apr.config import Config
from apr.model import Punctuator

@pytest.fixture(scope="session", autouse=True)
def cpu_threads():
    torch.set_num_threads(1)

@pytest.fixture
def tokenizer():
    vocabulary = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", ".", ",", "?", "hello", "world", "play", "##ing", "this", "is", "a", "test", "how", "are", "you", "yes", "no", "today"]
    raw = Tokenizer(models.WordPiece({word: i for i, word in enumerate(vocabulary)}, unk_token="[UNK]"))
    raw.normalizer = normalizers.Lowercase()
    raw.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    return PreTrainedTokenizerFast(tokenizer_object=raw, pad_token="[PAD]", unk_token="[UNK]", cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")

@pytest.fixture
def model(tokenizer):
    torch.manual_seed(42)
    c = Config(embedding_size=8, hidden_size=12, intermediate_size=24, num_heads=3, num_layers=6,
               vocab_size=len(tokenizer), dropout=0.1)
    return Punctuator(c, [2,3,4,5,6]).eval()
