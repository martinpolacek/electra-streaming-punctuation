import torch
import pytest
from apr.config import Config
from apr.pretraining import RTD, AdamWNoBias, CorpusSampler


def test_rtd_shared_embedding_loss_and_no_bias_adam(tokenizer):
    c = Config(embedding_size=8, hidden_size=16, intermediate_size=32, num_heads=2, num_layers=2, vocab_size=len(tokenizer))
    system = RTD(c, tokenizer)
    assert system.encoder.embeddings.word_embeddings.weight is system.generator.electra.embeddings.word_embeddings.weight
    ids = torch.tensor([[8,9,10,11,2,0]])
    loss, gen, disc = system(ids, ids != 0)
    assert torch.isfinite(loss) and torch.isfinite(gen) and torch.isfinite(disc)
    loss.backward()
    assert system.encoder.embeddings.word_embeddings.weight.grad is not None
    p = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = AdamWNoBias([p], lr=0.1, weight_decay=0.01)
    p.grad = torch.tensor([2.0])
    expected = 1 - 0.1*((0.1*2)/((0.001*4)**0.5+1e-6) + 0.01)
    optimizer.step()
    assert p.item() == pytest.approx(expected)


def test_sampler_sees_tail_and_resumes(tmp_path):
    (tmp_path / "sample.utf8").write_text(" ".join("w"+str(i) for i in range(3000)), encoding="utf-8")
    sampler = CorpusSampler(tmp_path, 1, 42)
    batches = [sampler.batch()[0] for _ in range(40)]
    assert any("w2999" in text for text in batches)
    state = sampler.state_dict()
    restored = CorpusSampler(tmp_path, 1, 999)
    restored.load_state_dict(state)
    assert sampler.batch() == restored.batch()
