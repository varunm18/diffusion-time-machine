"""CPU tests for date FiLM (per-block date scale-and-shift), on a tiny UNet made of real SEVA ResBlocks."""
import pytest
import torch
from torch import nn

layers = pytest.importorskip("seva.modules.layers", reason="SEVA not installed (scripts/setup_seva.sh)")

from timemachine.model.date_film import date_film_layers  # noqa: E402
from timemachine.model.dated_seva import DatedSGMWrapper  # noqa: E402
from timemachine.model.finetune import (  # noqa: E402
    FinetuneConfig, load_trainable_state_dict, prepare_for_finetuning, trainable_state_dict,
)
from timemachine.model.grad_checkpoint import enable_gradient_checkpointing  # noqa: E402

C = 32


class TinyResSeva(nn.Module):
    """SEVA's interface and block attributes, with two real ResBlocks."""

    model_channels = 8

    def __init__(self):
        super().__init__()
        self.time_embed = nn.Linear(1, 4 * self.model_channels)
        self.input_blocks = nn.ModuleList([layers.ResBlock(C, 4 * self.model_channels, C, 6, 0.0)])
        self.middle_block = layers.ResBlock(C, 4 * self.model_channels, C, 6, 0.0)
        self.output_blocks = nn.ModuleList()

    def forward(self, x, t, y, dense_y, num_frames=None):
        emb = self.time_embed(t[:, None].float())
        h = self.input_blocks[0](x, emb, dense_y)
        return self.middle_block(h, emb, dense_y)


def _inputs(n=4):
    torch.manual_seed(1)
    x, t = torch.randn(n, C - 2, 4, 4), torch.tensor([10, 10, 500, 500][:n])
    c = {"crossattn": torch.zeros(n, 1, 8), "dense_vector": torch.randn(n, 6, 4, 4),
         "concat": torch.zeros(n, 2, 4, 4)}
    return x, t, c


def _model(perturb: bool = False) -> DatedSGMWrapper:
    torch.manual_seed(0)
    model = DatedSGMWrapper(TinyResSeva())
    model.add_date_film(16)
    if perturb:
        for layer in date_film_layers(model.module).values():
            nn.init.normal_(layer.film.weight, std=0.1)
    return model


def test_date_film_is_identity_at_init_and_uses_dates_after_training():
    x, t, c = _inputs()
    years = torch.tensor([1860.0, 1860.0, 2020.0, 2020.0])
    model = _model()
    assert len(date_film_layers(model.module)) == 2
    reference = model(x, t, c)                                         # no dates: FiLM is skipped
    torch.testing.assert_close(model(x, t, c, date=years), reference)  # zero-init: identical

    model = _model(perturb=True)
    a = model(x, t, c, date=years)
    b = model(x, t, c, date=torch.full((4,), 1900.0))
    unknown = model(x, t, c, date=years, date_valid=torch.tensor([True, True, False, False]))
    assert not torch.allclose(a, b) and not torch.allclose(a, unknown)
    assert model.date_module.state.features is not None                # kept for the backward pass
    torch.testing.assert_close(model(x, t, c), reference)              # plain call clears it again
    assert model.date_module.state.features is None


def test_date_film_handles_the_doubled_cfg_batch():
    x, t, c = _inputs()
    model = _model(perturb=True)
    years = torch.tensor([1860.0, 2020.0])
    with model.dates(years):
        out = model(x, t, c)                                           # 4 views, 2 dates: repeated
    expected = model(x, t, c, date=years.repeat(2))
    torch.testing.assert_close(out, expected)


def test_date_film_gradients_survive_checkpointing():
    """Backward runs after the dates context has closed; the recomputed blocks must still see the dates."""
    x, t, c = _inputs()
    years = torch.tensor([1860.0, 1900.0, 2020.0, 2021.0])
    grads = []
    for checkpointed in (False, True):
        model = _model(perturb=True)
        if checkpointed:
            enable_gradient_checkpointing(model.module)
        model.train()
        valid = torch.tensor([True, True, True, False])               # one unknown date: its vector learns too
        model(x, t, c, date=years, date_valid=valid).pow(2).mean().backward()
        grads.append([p.grad.clone() for p in model.film_parameters()])
    for plain, ckpt in zip(*grads):
        torch.testing.assert_close(plain, ckpt)
    assert all(g.abs().sum() > 0 for g in grads[1])                    # encoder and every block learn


def test_finetune_with_film_trains_only_film_and_round_trips():
    x, t, c = _inputs()
    years = torch.tensor([1860.0, 1860.0, 2020.0, 2020.0])
    torch.manual_seed(0)
    model = DatedSGMWrapper(TinyResSeva())
    groups = prepare_for_finetuning(model, FinetuneConfig(lora_rank=0, date_injection="film", film_dim=16))
    assert groups["film"] and not groups["date"] and not groups["lora"]
    trainable = {id(p) for p in groups["film"]}
    assert all(p.requires_grad == (id(p) in trainable) for p in model.parameters())
    for layer in date_film_layers(model.module).values():
        nn.init.normal_(layer.film.weight, std=0.1)
    model.eval()
    expected = model(x, t, c, date=years)

    torch.manual_seed(1)
    fresh = DatedSGMWrapper(TinyResSeva())
    fresh.module.load_state_dict({k.replace(".conv.", ".") if ".in_layers.2.conv." in k else k: v
                                  for k, v in model.module.state_dict().items() if ".film." not in k
                                  and "film_encoder" not in k}, strict=False)
    prepare_for_finetuning(fresh, FinetuneConfig(lora_rank=0, date_injection="film", film_dim=16))
    load_trainable_state_dict(fresh, trainable_state_dict(model))
    fresh.eval()
    torch.testing.assert_close(fresh(x, t, c, date=years), expected)


def test_finetune_config_rejects_unknown_injection():
    with pytest.raises(ValueError):
        FinetuneConfig(date_injection="crossattn")
