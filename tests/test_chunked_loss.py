"""The chunked loss equals the full-logits causal-LM loss, value and gradients, without building the logits."""
import pytest
import torch
from torch import nn

import train.model as model_module
from train.model import DeepSeekOCR


class StubHF(nn.Module):
    """The two pieces the chunked loss touches: `.model(**kw)` -> (hidden,...) and `.lm_head`."""

    def __init__(self, vocab=50, hidden=8):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.mix = nn.Linear(hidden, hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        self.model = lambda input_ids, **_: (torch.tanh(self.mix(self.embed(input_ids))),)
        self.full_calls = 0

    def forward(self, input_ids, labels=None, **_):                 # the reference: logits for every position
        self.full_calls += 1
        logits = self.lm_head(self.model(input_ids)[0])
        loss = nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                                           labels[:, 1:].reshape(-1), ignore_index=-100)
        return type("Out", (), {"loss": loss, "logits": logits})()


def _wrapper():
    torch.manual_seed(0)
    hf = StubHF()
    wrapper = DeepSeekOCR.__new__(DeepSeekOCR)
    nn.Module.__init__(wrapper)
    wrapper.model, wrapper.chunked_loss = hf, True
    return wrapper, hf


def _batch():
    ids = torch.randint(0, 50, (3, 40))
    labels = ids.clone()
    labels[:, :15] = -100                                           # the prompt is not supervised
    labels[1, 30:] = -100                                           # a shorter answer (right padding)
    return {"input_ids": ids, "labels": labels}


def test_value_and_gradients_match_the_full_logits_loss(monkeypatch):
    monkeypatch.setattr(model_module, "LOSS_CHUNK", 7)              # several chunks, the last one partial
    wrapper, hf = _wrapper()
    batch = _batch()

    wrapper.chunked_loss = False
    reference = wrapper(**batch)
    reference.loss.backward()
    expected = {n: p.grad.clone() for n, p in hf.named_parameters()}
    hf.zero_grad()

    wrapper.chunked_loss = True
    chunked = wrapper(**batch)
    chunked.loss.backward()
    assert chunked.logits is None                                    # the full logits were never kept
    assert torch.allclose(chunked.loss, reference.loss, atol=1e-6)
    for name, parameter in hf.named_parameters():
        assert torch.allclose(parameter.grad, expected[name], atol=1e-6), name


def test_an_example_without_labels_or_with_positional_arguments_takes_the_normal_path():
    wrapper, hf = _wrapper()
    wrapper(input_ids=torch.randint(0, 50, (1, 10)), labels=torch.randint(0, 50, (1, 10)))  # chunked: no full call
    assert hf.full_calls == 0
    wrapper.chunked_loss = False
    wrapper(input_ids=torch.randint(0, 50, (1, 10)), labels=torch.randint(0, 50, (1, 10)))
    assert hf.full_calls == 1


def test_a_batch_with_nothing_supervised_does_not_divide_by_zero():
    wrapper, _ = _wrapper()
    ids = torch.randint(0, 50, (2, 12))
    out = wrapper(input_ids=ids, labels=torch.full_like(ids, -100))
    assert float(out.loss) == 0.0
