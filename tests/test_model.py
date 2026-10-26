"""CPU tests for the date-conditioning modules (real SEVA needs a GPU: see scripts/seva_baseline.py)."""
import pytest
import torch
from torch import nn

seva = pytest.importorskip("seva.model", reason="SEVA not installed (scripts/setup_seva.sh)")

from timemachine.model.date_embedding import DateEmbedding  # noqa: E402
from timemachine.model.dated_seva import DatedSGMWrapper, TimeEmbedWithDate  # noqa: E402


class TinySeva(nn.Module):
    """Stand-in with SEVA's interface: forward adds time_embed(t) to the input channels."""

    model_channels = 8

    def __init__(self):
        super().__init__()
        self.time_embed = nn.Linear(1, 4 * self.model_channels)
        self.proj = nn.Linear(4 * self.model_channels, 4)

    def forward(self, x, t, y, dense_y, num_frames=None):
        emb = self.time_embed(t[:, None].float())
        return x[:, :4] + self.proj(emb)[:, :, None, None]


def test_date_embedding_starts_at_zero_and_handles_unknown():
    emb = DateEmbedding(out_dim=16)
    years = torch.tensor([1850.0, 1925.5, 2019.3])
    assert torch.equal(emb(years), torch.zeros(3, 16))
    nn.init.normal_(emb.mlp[-1].weight)
    known = emb(years, torch.tensor([True, True, True]))
    unknown = emb(years, torch.tensor([True, False, True]))
    assert not torch.allclose(known[1], unknown[1]) and torch.allclose(known[0], unknown[0])
    assert not torch.allclose(known[0], known[2])   # different years, different embeddings


def test_time_embed_with_date_repeats_for_cfg_batch():
    base = nn.Linear(3, 16)
    date = DateEmbedding(out_dim=16)
    nn.init.normal_(date.mlp[-1].weight)
    hook = TimeEmbedWithDate(base, date)
    t = torch.randn(4, 3)
    assert torch.equal(hook(t[:2]), base(t[:2]))                    # no dates set -> plain SEVA
    hook.year, hook.valid = torch.tensor([1900.0, 2000.0]), torch.tensor([True, True])
    out = hook(t)                                                   # CFG: 2x the dates
    expected = base(t) + date(torch.tensor([1900.0, 2000.0, 1900.0, 2000.0]))
    torch.testing.assert_close(out, expected)
    with pytest.raises(ValueError):
        hook(t[:3])


def test_dated_wrapper_is_identity_at_init_and_uses_dates_after_training():
    torch.manual_seed(0)
    plain = TinySeva()
    dated = DatedSGMWrapper(TinySeva())
    dated.module.load_state_dict({k.replace("time_embed.", "time_embed.time_embed.") if k.startswith("time_embed")
                                  else k: v for k, v in plain.state_dict().items()}, strict=False)
    x, t = torch.randn(4, 4, 2, 2), torch.tensor([10, 10, 500, 500])
    c = {"crossattn": torch.zeros(4, 1, 8), "dense_vector": torch.zeros(4, 6, 2, 2), "concat": torch.zeros(4, 7, 2, 2)}
    reference = seva.SGMWrapper(plain)(x, t, c)
    years = torch.tensor([1860.0, 1860.0, 2020.0, 2020.0])
    torch.testing.assert_close(dated(x, t, c, date=years), reference)          # zero-init: identical
    nn.init.normal_(dated.date_module.date_embedding.mlp[-1].weight)
    with dated.dates(years):
        a = dated(x, t, c)
    b = dated(x, t, c, date=torch.full((4,), 1900.0))
    assert not torch.allclose(a, b)                                            # dates now matter
    assert dated.date_module.year is None                                      # context cleaned up
    assert all(p.requires_grad for p in dated.date_parameters())
