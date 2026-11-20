"""CPU tests for MVGenMaster's date conditioner and adapters (``timemachine.mvgen.date`` and ``.model``).

They run in MVGenMaster's environment, from its checkout (see ``timemachine/mvgen/__init__.py``), on
two small stand-ins built from MVGenMaster's own classes; the released UNet is never loaded:

* ``TinyUNet``: two real ResBlocks behind the UNet's calling convention. Its views do not interact,
  so a test can tell which view a date reached.
* ``_tiny_model()``: a small ``UNet2DConditionModel`` with the released model's layout (no
  cross-attention, a domain embedding), which exercises MVGenMaster's own forward pass, gradient
  checkpointing and layer names.
"""
import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

my_diffusers = pytest.importorskip("my_diffusers", reason="runs in MVGenMaster's environment, from its checkout")

# Under pytest the repo root is on the import path (its conftest.py), so ``src`` is the repo's own ``src``
# package and hides MVGenMaster's ``src`` folder, which ``timemachine.mvgen`` imports from. Let the package
# search that folder too (outside pytest the repo root is not on the path and nothing needs doing).
import src  # noqa: E402

MVGEN_SRC = str(Path(my_diffusers.__file__).resolve().parents[1] / "src")
if MVGEN_SRC not in list(src.__path__):
    src.__path__.append(MVGEN_SRC)

from diffusers import AutoencoderKL  # noqa: E402
from easydict import EasyDict  # noqa: E402
from my_diffusers.models import UNet2DConditionModel  # noqa: E402
from my_diffusers.models.attention import BasicTransformerBlock  # noqa: E402
from my_diffusers.models.resnet import ResnetBlock2D  # noqa: E402

from timemachine.model.date_embedding import fourier_date_features  # noqa: E402
from timemachine.model.lora import LoRALinear, add_lora, lora_layers  # noqa: E402
from timemachine.mvgen import model as mvgen_model  # noqa: E402
from timemachine.mvgen.date import DateConditioner, DateEmbedding, dated_blocks, temb_width  # noqa: E402
from timemachine.mvgen.model import (  # noqa: E402
    DEFAULT_LORA_TARGETS, TIME_PROJ_TARGET, FinetuneConfig, MVGen, assert_exact_matmul, load_finetuned, load_mvgen,
    load_trainable_state_dict, prepare_for_finetuning, save_checkpoint, trainable_state_dict,
)

WIDTH = 32      # TinyUNet's timestep-embedding width
FRAMES = 8
YEARS = torch.tensor([1855.0, 1890.0, 1925.0, 1960.0, 1995.0, 2010.0, 2018.0, 2021.0])
READER_LORA_TARGETS = (*DEFAULT_LORA_TARGETS, TIME_PROJ_TARGET)     # LoRA also on the layers that read the date


# ----------------------------------------------------------------------------- stand-ins

class TinyUNet(nn.Module):
    """MVGenMaster's calling convention around two real ResBlocks.

    ``sample`` [B, F, C, h, w] and ``timestep`` [B] -> [(B*F), C, h, w]. As in the real UNet, the
    timestep embedding is computed per batch element and repeated per view, and (when
    ``checkpointed``) every ResBlock call is checkpointed on its own, non-reentrant.
    """

    def __init__(self, channels: int = 8):
        super().__init__()
        self.time_embedding = nn.Linear(1, WIDTH)
        self.resnets = nn.ModuleList(ResnetBlock2D(in_channels=channels, temb_channels=WIDTH, groups=4)
                                     for _ in range(2))
        self.checkpointed = False

    def forward(self, sample: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        temb = self.time_embedding(timestep[:, None] / 1000.0).repeat_interleave(sample.shape[1], dim=0)
        hidden = sample.flatten(0, 1)
        for resnet in self.resnets:
            if self.checkpointed:
                hidden = checkpoint(resnet, hidden, temb, use_reentrant=False)
            else:
                hidden = resnet(hidden, temb)
        return hidden


def _tiny(perturbed: bool = False, checkpointed: bool = False) -> tuple[TinyUNet, DateConditioner]:
    """A TinyUNet with a date conditioner; ``perturbed``: as after some training (last layer not zero)."""
    torch.manual_seed(0)
    unet = TinyUNet()
    unet.checkpointed = checkpointed
    conditioner = DateConditioner(unet, DateEmbedding(WIDTH))
    if perturbed:
        nn.init.normal_(conditioner.embedding.mlp[-1].weight, std=0.5)
    return unet, conditioner


def _inputs(batch: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """``batch`` identical samples of 8 views, and their timesteps."""
    torch.manual_seed(1)
    return torch.randn(1, FRAMES, 8, 4, 4).repeat(batch, 1, 1, 1, 1), torch.full((batch,), 500)


def _changed_views(a: torch.Tensor, b: torch.Tensor) -> list[bool]:
    return (~torch.isclose(a, b, rtol=0, atol=1e-6)).flatten(1).any(1).tolist()


def _rms(x: torch.Tensor) -> float:
    return x.detach().float().pow(2).mean().sqrt().item()


def _tiny_model() -> MVGen:
    """A small model of MVGenMaster's own UNet class, laid out like the released one."""
    torch.manual_seed(0)
    unet = UNet2DConditionModel(
        sample_size=8, block_out_channels=(32, 64), layers_per_block=1, norm_num_groups=8,
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"), up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
        cross_attention_dim=None, attention_head_dim=4, use_linear_projection=True,
        class_embed_type="zero_init", num_class_embeds=3)
    unet.use_rope = False       # the real loader sets it (from_pretrained with model_cfg)
    vae = AutoencoderKL(block_out_channels=(8,), norm_num_groups=4, latent_channels=4)
    return MVGen(unet=unet.eval(), vae=vae.eval(), config=EasyDict(), scheduler=None)


def _predict(model: MVGen, batch: int = 1) -> torch.Tensor:
    """One UNet call as in training and sampling: ``batch`` samples of 3 views, timesteps 100, 900, ..."""
    torch.manual_seed(1)
    sample = torch.randn(batch, 3, 4, 8, 8)
    timesteps = torch.tensor([100, 900])[:batch]
    return model.unet(sample, timesteps, None, class_labels=torch.zeros(batch, dtype=torch.long), return_dict=False)[0]


def _train_adapters(model: MVGen) -> None:
    """Make the zero-initialised layers non-zero, as after some training."""
    torch.manual_seed(2)
    if model.date is not None:
        nn.init.normal_(model.date.embedding.mlp[-1].weight, std=0.5)
    for layer in lora_layers(model.unet).values():
        nn.init.normal_(layer.up.weight, std=0.2)


# ----------------------------------------------------------------------------- date.py

def test_date_embedding_is_zero_at_init_and_separates_dates():
    torch.manual_seed(0)
    embedding = DateEmbedding(out_dim=WIDTH)
    assert embedding.unknown.shape == (17,)         # the default 8 octaves: 8 sines, 8 cosines and the year itself
    assert DateEmbedding(out_dim=WIDTH, n_freqs=4).unknown.shape == (9,)
    assert FinetuneConfig().date_freqs == 8
    year = torch.tensor([1890.0, 1890.0, 2015.0, 2015.0])
    valid = torch.tensor([True, True, True, False])
    assert torch.count_nonzero(embedding(year, valid)) == 0

    nn.init.normal_(embedding.mlp[-1].weight, std=0.5)
    out = embedding(year, valid)
    assert out.shape == (4, WIDTH)
    assert torch.equal(out[0], out[1])                                      # same date, same vector
    assert not torch.allclose(out[0], out[2])                               # another date
    assert not torch.allclose(out[2], out[3])                               # unknown is not a date
    unknown = embedding(torch.tensor([1700.0, float("nan")]), torch.tensor([False, False]))
    torch.testing.assert_close(unknown, out[3].expand(2, -1))               # an unknown date ignores the year


def test_default_octaves_tell_the_months_around_the_fire_apart():
    """Why the default is 8 octaves: to 4 octaves, January and July 2019 are almost one input."""
    def distance(n_freqs: int) -> float:
        embedding = DateEmbedding(out_dim=WIDTH, n_freqs=n_freqs)
        before, after = fourier_date_features(torch.tensor([2019.05, 2019.55]), torch.ones(2, dtype=torch.bool),
                                              embedding.freqs, embedding.unknown)
        return (before - after).norm().item()

    assert distance(4) < 0.2 and distance(8) > 1.5


def test_conditioner_is_an_exact_noop_at_init():
    torch.manual_seed(0)
    unet = TinyUNet()
    sample, t = _inputs()
    base = unet(sample, t)
    keys = set(unet.state_dict())
    conditioner = DateConditioner(unet, DateEmbedding(WIDTH))
    assert set(unet.state_dict()) == keys                                   # nothing is added to the UNet
    assert {name.split(".")[0] for name, _ in conditioner.named_parameters()} == {"embedding"}
    assert torch.equal(unet(sample, t), base)                               # no dates set
    valid = torch.tensor([True] * 6 + [False] * 2)
    with conditioner.dates(YEARS, valid):
        assert torch.equal(unet(sample, t), base)                           # zero-init: bit for bit


def test_each_view_gets_its_own_date():
    unet, conditioner = _tiny(perturbed=True)
    sample, t = _inputs()
    base = unet(sample, t)

    def run(year, valid=None):
        with conditioner.dates(year, valid):
            return unet(sample, t)

    dated = run(YEARS)
    assert all(_changed_views(dated, base))
    assert torch.equal(run(YEARS), dated)
    other = YEARS.clone()
    other[2] = 1700.0
    assert _changed_views(run(other), dated) == [False, False, True, False, False, False, False, False]
    valid = torch.ones(FRAMES, dtype=torch.bool)
    valid[5] = False
    assert _changed_views(run(YEARS, valid), dated) == [False, False, False, False, False, True, False, False]
    assert torch.equal(unet(sample, t), base)                               # outside dates(): the base model


def test_dates_fill_a_guidance_doubled_batch():
    """Two samples x eight views: the dates repeat per sample; the unconditional half can be undated."""
    unet, conditioner = _tiny(perturbed=True)
    sample, t = _inputs(batch=2)
    seen = []
    unet.resnets[1].register_forward_pre_hook(lambda module, args: seen.append(args[1]))    # after the date hook

    def run(year=None, valid=None, uncond="same", inputs=(sample, t)):
        """(output, the temb the second ResBlock received)."""
        seen.clear()
        if year is None:
            return unet(*inputs), seen[0]
        with conditioner.dates(year, valid, uncond):
            return unet(*inputs), seen[0]

    valid = torch.ones(FRAMES, dtype=torch.bool)
    valid[3] = False
    nothing_known = torch.zeros(FRAMES, dtype=torch.bool)
    dated, undated = conditioner.embedding(YEARS, valid), conditioner.embedding(YEARS, nothing_known)
    plain_temb = run()[1]
    assert plain_temb.shape == (2 * FRAMES, WIDTH)
    torch.testing.assert_close(run(YEARS, valid)[1] - plain_temb, torch.cat([dated, dated]))
    out, temb = run(YEARS, valid, uncond="unknown")
    torch.testing.assert_close(temb - plain_temb, torch.cat([undated, dated]))             # unconditional half first
    one = (sample[:1], t[:1])
    torch.testing.assert_close(out, torch.cat([run(YEARS, nothing_known, inputs=one)[0],
                                               run(YEARS, valid, inputs=one)[0]]))
    per_view = torch.cat([YEARS, YEARS.flip(0)])                                           # one date per view
    torch.testing.assert_close(run(per_view)[1] - plain_temb,
                               conditioner.embedding(per_view, torch.ones(2 * FRAMES, dtype=torch.bool)))


def test_dates_are_validated():
    unet, conditioner = _tiny(perturbed=True)
    sample, t = _inputs()
    base = unet(sample, t)
    with pytest.raises(ValueError, match="uncond"):
        with conditioner.dates(YEARS, uncond="none"):
            pass
    with pytest.raises(ValueError, match="equal length"):
        with conditioner.dates(YEARS, torch.ones(3, dtype=torch.bool)):
            pass
    with pytest.raises(ValueError, match="cannot fill"):
        with conditioner.dates(YEARS[:3]):                                  # 3 dates for 8 views
            unet(sample, t)
    assert torch.equal(unet(sample, t), base)                               # the failed block left no dates behind


def test_embedding_is_computed_once_per_forward_pass():
    unet, conditioner = _tiny(perturbed=True, checkpointed=True)
    sample, t = _inputs()
    calls = []
    conditioner.embedding.register_forward_hook(lambda *_: calls.append(1))
    with conditioner.dates(YEARS):
        out = unet(sample, t)
    assert len(calls) == 1                          # two ResBlocks, one embedding
    out.pow(2).mean().backward()                    # checkpointing runs both ResBlocks again
    assert len(calls) == 1
    with conditioner.dates(YEARS):
        unet(sample, t)
        unet(sample, t)
    assert len(calls) == 3
    unet(sample, t)                                 # undated
    assert len(calls) == 3


def test_last_stats_are_the_sizes_in_the_latest_pass():
    unet, conditioner = _tiny()
    sample, t = _inputs(batch=2)
    seen = []
    unet.resnets[0].register_forward_pre_hook(lambda module, args: seen.append(args[1]))
    assert conditioner.last_stats is None                                   # no pass yet
    unet(sample, t)
    temb_rms = _rms(seen[-1])                                               # timestep embedding alone
    assert conditioner.last_stats is None                                   # an undated pass
    with conditioner.dates(YEARS):
        unet(sample, t)
    assert conditioner.last_stats == {"date_rms": 0.0, "temb_rms": pytest.approx(temb_rms)}     # zero-init

    nn.init.normal_(conditioner.embedding.mlp[-1].weight, std=0.5)          # as after some training
    valid = torch.ones(FRAMES, dtype=torch.bool)
    valid[3] = False
    with conditioner.dates(YEARS, valid, uncond="unknown"):
        unet(sample, t)
    stats = conditioner.last_stats
    added = torch.cat([conditioner.embedding(YEARS, torch.zeros(FRAMES, dtype=torch.bool)),
                       conditioner.embedding(YEARS, valid)])                # every row of the doubled batch counts
    assert set(stats) == {"date_rms", "temb_rms"} and all(type(value) is float for value in stats.values())
    assert stats["date_rms"] == pytest.approx(_rms(added)) and stats["date_rms"] > 0
    assert stats["temb_rms"] == pytest.approx(temb_rms)                     # temb before the date was added
    with pytest.raises(ValueError, match="cannot fill"), conditioner.dates(YEARS[:3]):
        unet(sample, t)
    assert conditioner.last_stats is None                                   # a failed pass reports nothing
    with conditioner.dates(YEARS):
        unet(sample, t)
    unet(sample, t)
    assert conditioner.last_stats is None                                   # undated again


def test_last_stats_are_taken_once_per_pass_and_never_in_backward():
    unet, conditioner = _tiny(perturbed=True, checkpointed=True)
    sample, t = _inputs()
    runs, after_first_block = [], []
    unet.resnets[0].register_forward_pre_hook(lambda *_: runs.append(1))
    unet.resnets[0].register_forward_hook(lambda *_: after_first_block.append(conditioner.last_stats))
    with conditioner.dates(YEARS):
        out = unet(sample, t)
    first = conditioner.last_stats
    assert first is after_first_block[0]                                    # the second ResBlock left them alone
    with torch.no_grad(), conditioner.dates(YEARS, torch.zeros(FRAMES, dtype=torch.bool)):
        unet(sample, t)
    second = conditioner.last_stats
    assert second["date_rms"] != first["date_rms"] and second["temb_rms"] == first["temb_rms"]
    out.pow(2).mean().backward()                    # checkpointing re-runs the first pass's ResBlocks
    assert len(runs) == 3 and conditioner.last_stats is second


def _date_gradient(checkpointed: bool, other_pass: str | None = None) -> torch.Tensor:
    """Gradient of a dated pass w.r.t. the date parameters; optionally with a gradient-free pass at other dates."""
    unet, conditioner = _tiny(perturbed=True, checkpointed=checkpointed)
    sample, t = _inputs()
    valid = torch.ones(FRAMES, dtype=torch.bool)
    valid[0] = False

    def reference_pass():
        with torch.no_grad(), conditioner.dates(YEARS.flip(0)):
            unet(sample, t)

    if other_pass == "before":
        reference_pass()
    with conditioner.dates(YEARS, valid):
        out = unet(sample, t)
    if other_pass == "between":
        reference_pass()
    out.pow(2).mean().backward()                    # outside dates(), as in training
    assert all(p.grad.abs().sum() > 0 for p in conditioner.parameters())
    return torch.cat([p.grad.flatten() for p in conditioner.parameters()])


def test_checkpointed_gradients_equal_plain_gradients():
    plain = _date_gradient(checkpointed=False)
    torch.testing.assert_close(_date_gradient(checkpointed=True), plain, rtol=1e-6, atol=1e-9)
    # a gradient-free pass at other dates BEFORE the forward pass is harmless
    torch.testing.assert_close(_date_gradient(checkpointed=True, other_pass="before"), plain, rtol=1e-6, atol=1e-9)
    # ... and so is one BETWEEN the forward pass and its backward(): each pass's embedding is bound to its temb
    torch.testing.assert_close(_date_gradient(checkpointed=True, other_pass="between"), plain, rtol=1e-6, atol=1e-9)
    torch.testing.assert_close(_date_gradient(checkpointed=False, other_pass="between"), plain, rtol=1e-6, atol=1e-9)


def test_a_rerun_that_lost_its_temb_raises_instead_of_guessing():
    unet, conditioner = _tiny(perturbed=True)
    sample, t = _inputs()
    with conditioner.dates(YEARS):
        unet(sample, t)
    stray_temb = torch.zeros(FRAMES, WIDTH)                                 # not created by a UNet forward pass
    with pytest.raises(RuntimeError, match="outside a UNet forward pass"):
        unet.resnets[0](sample.flatten(0, 1), stray_temb)


def test_two_dated_passes_before_one_backward_each_keep_their_dates():
    """What a ranking loss does: two passes with gradients, one backward()."""
    def gradient(checkpointed: bool) -> torch.Tensor:
        unet, conditioner = _tiny(perturbed=True, checkpointed=checkpointed)
        sample, t = _inputs()
        with conditioner.dates(YEARS):
            first = unet(sample, t)
        with conditioner.dates(YEARS.flip(0)):
            second = unet(sample, t)
        (first.pow(2).mean() + 2.0 * second.pow(3).mean()).backward()
        return torch.cat([p.grad.flatten() for p in conditioner.parameters()])

    torch.testing.assert_close(gradient(checkpointed=True), gradient(checkpointed=False), rtol=1e-6, atol=1e-9)


def test_remove_detaches_the_hooks():
    unet, conditioner = _tiny(perturbed=True)
    sample, t = _inputs()
    base = unet(sample, t)
    with conditioner.dates(YEARS):
        assert not torch.allclose(unet(sample, t), base)
    assert conditioner.last_stats is not None
    conditioner.remove()
    assert conditioner.last_stats is None
    with conditioner.dates(YEARS):
        assert torch.equal(unet(sample, t), base)
    assert conditioner.last_stats is None


def test_conditioner_needs_matching_resblocks():
    with pytest.raises(ValueError, match="width"):
        DateConditioner(TinyUNet(), DateEmbedding(WIDTH + 1))
    with pytest.raises(ValueError, match="ResnetBlock2D"):
        DateConditioner(nn.Linear(2, 2), DateEmbedding(WIDTH))


def test_embedding_stays_fp32_under_autocast():
    """bf16 autocast: the date MLP still runs in fp32, and its output is cast to temb's dtype for the addition."""
    unet, conditioner = _tiny(perturbed=True)
    sample, t = _inputs()
    embedding_dtypes, temb_dtypes = [], []
    conditioner.embedding.register_forward_hook(lambda module, args, output: embedding_dtypes.append(output.dtype))
    unet.resnets[0].register_forward_pre_hook(lambda module, args: temb_dtypes.append(args[1].dtype))
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16), conditioner.dates(YEARS):
        out = unet(sample, t)
    assert embedding_dtypes == [torch.float32] and temb_dtypes == [torch.bfloat16]     # TinyUNet's temb is bf16 here
    rounded_stats = conditioner.last_stats
    with torch.no_grad():
        base = unet(sample, t)
        with conditioner.dates(YEARS):
            exact = unet(sample, t)
    assert (out.float() - exact).abs().max() < 0.2 * (exact - base).abs().max()     # the dated output, bf16-rounded
    assert rounded_stats["date_rms"] == conditioner.last_stats["date_rms"]          # the fp32 embedding's size
    assert rounded_stats["temb_rms"] == pytest.approx(conditioner.last_stats["temb_rms"], rel=0.01)


# ----------------------------------------------------------------------------- model.py

def test_default_lora_targets_are_the_transformer_block_linears():
    unet = _tiny_model().unet
    blocks = [name for name, m in unet.named_modules() if isinstance(m, BasicTransformerBlock)]
    expected = {name for name, m in unet.named_modules()
                if isinstance(m, nn.Linear) and any(name.startswith(block + ".") for block in blocks)}
    assert len(blocks) == 4 and len(expected) == 6 * len(blocks)
    assert all(unet.get_submodule(block).attn2 is None for block in blocks)      # no cross-attention, as released
    assert set(add_lora(unet, DEFAULT_LORA_TARGETS, rank=2)) == expected
    others = [m for name, m in unet.named_modules() if "time_emb" in name or name.endswith(("proj_in", "proj_out"))]
    assert others and not any(isinstance(m, LoRALinear) for m in others)


def test_prepare_for_finetuning_is_a_noop_and_trains_only_the_adapters():
    model = _tiny_model()
    valid = torch.tensor([True, True, False])
    with torch.no_grad():
        base = _predict(model, batch=2)
    groups = prepare_for_finetuning(model, FinetuneConfig(lora_rank=2))
    assert len(groups["date"]) == 5 and len(groups["lora"]) == 2 * 24           # 4 transformer blocks x 6 linears
    assert model.unet.is_gradient_checkpointing
    trainable = {id(p) for group in groups.values() for p in group}
    assert {id(p) for p in model.unet.parameters() if p.requires_grad} | {id(p) for p in model.date.parameters()} \
        == trainable
    assert not any(p.requires_grad for p in model.vae.parameters())
    with torch.no_grad(), model.dates(YEARS[:3], valid, uncond="unknown"):
        assert torch.equal(_predict(model, batch=2), base)                      # step 0: the base model, bit for bit

    _train_adapters(model)
    with torch.no_grad():
        undated = _predict(model, batch=2)
        with model.dates(YEARS[:3], valid):
            dated = _predict(model, batch=2)
    assert not torch.allclose(undated, base) and not torch.allclose(dated, undated)
    with pytest.raises(ValueError, match="already"):
        prepare_for_finetuning(model, FinetuneConfig())


def test_dates_reach_the_right_views_of_the_real_unet():
    """MVGenMaster's own forward pass: temb rows are views, batch elements one after the other (uncond first)."""
    model = _tiny_model()
    prepare_for_finetuning(model, FinetuneConfig())
    _train_adapters(model)
    seen = []
    model.unet.mid_block.resnets[0].register_forward_pre_hook(lambda module, args: seen.append(args[1]))
    years, valid = YEARS[:3], torch.tensor([True, True, False])
    assert model.date_stats() is None
    with torch.no_grad():
        _predict(model, batch=2)
        assert model.date_stats() is None           # an undated pass
        with model.dates(years, valid, uncond="unknown"):
            _predict(model, batch=2)
        stats = model.date_stats()
        with torch.autocast("cpu", dtype=torch.bfloat16), model.dates(years, valid):
            _predict(model, batch=2)
    plain, dated, autocast = seen
    assert plain.shape == (6, 128) and torch.equal(plain[0], plain[2]) and not torch.equal(plain[2], plain[3])
    expected = torch.cat([model.date.embedding(years, torch.zeros(3, dtype=torch.bool)),
                          model.date.embedding(years, valid)])
    torch.testing.assert_close(dated - plain, expected)
    assert autocast.dtype == torch.float32          # the fp32 domain embedding keeps temb in fp32 under bf16 autocast
    assert stats == {"date_rms": pytest.approx(_rms(expected)), "temb_rms": pytest.approx(_rms(plain))}
    assert model.date_stats() is model.date.last_stats                          # now those of the bf16 pass
    assert model.date_stats()["temb_rms"] == pytest.approx(stats["temb_rms"], rel=0.01)


def _adapter_gradient(checkpointing: bool) -> tuple[torch.Tensor, int]:
    """Gradient w.r.t. date and LoRA parameters through MVGenMaster's forward pass, and the ResBlock calls made."""
    model = _tiny_model()
    groups = prepare_for_finetuning(model, FinetuneConfig(lora_rank=2))
    _train_adapters(model)
    model.unet.train(checkpointing)                 # MVGenMaster's blocks only checkpoint in training mode
    calls = []
    for block in dated_blocks(model.unet):
        block.register_forward_pre_hook(lambda module, args: calls.append(1))
    with torch.no_grad(), model.dates(YEARS[3:6]):  # a gradient-free reference pass first
        _predict(model)
    calls.clear()
    with model.dates(YEARS[:3], torch.tensor([True, True, False])):
        out = _predict(model)
    out.pow(2).mean().backward()
    parameters = groups["date"] + groups["lora"]
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in parameters)
    return torch.cat([p.grad.flatten() for p in parameters]), len(calls)


def test_gradients_survive_mvgenmasters_gradient_checkpointing():
    plain, plain_calls = _adapter_gradient(checkpointing=False)
    checkpointed, checkpointed_calls = _adapter_gradient(checkpointing=True)
    assert plain_calls == 8                         # the tiny UNet's ResBlocks, once each
    assert checkpointed_calls > plain_calls         # ... and again in backward(), where the hooks ran with them
    torch.testing.assert_close(checkpointed, plain, rtol=1e-5, atol=1e-8)


def test_trainable_state_round_trip(tmp_path, monkeypatch):
    cfg = FinetuneConfig(lora_rank=2, lora_alpha=4.0)
    model = _tiny_model()
    prepare_for_finetuning(model, cfg)
    _train_adapters(model)
    years, valid = YEARS[:3], torch.tensor([True, False, True])
    with torch.no_grad(), model.dates(years, valid):
        trained = _predict(model)

    state = trainable_state_dict(model)
    assert set(state) == {"date", "lora"} and len(state["date"]) == 5 and len(state["lora"]) == 2 * 24
    assert all(not v.requires_grad and v.device.type == "cpu" for part in state.values() for v in part.values())
    kept = state["date"]["embedding.mlp.2.weight"].clone()
    with torch.no_grad():
        model.date.embedding.mlp[-1].weight.zero_()                             # the state holds copies
    assert torch.equal(state["date"]["embedding.mlp.2.weight"], kept)
    load_trainable_state_dict(model, state)

    path = tmp_path / "run" / "step_7.pt"
    save_checkpoint(path, model, cfg, step=7, extra={"note": "kept"})
    assert [p.name for p in path.parent.iterdir()] == ["step_7.pt"]             # no temporary file left
    with pytest.raises(ValueError, match="redefine"):
        save_checkpoint(path, model, cfg, step=8, extra={"step": 9})

    monkeypatch.setattr(mvgen_model, "load_mvgen", lambda model_dir, device: _tiny_model())
    restored, ckpt = load_finetuned(path, device="cpu")
    assert ckpt["step"] == 7 and ckpt["note"] == "kept" and FinetuneConfig(**ckpt["finetune_config"]) == cfg
    assert not restored.unet.training
    with torch.no_grad(), restored.dates(years, valid):
        torch.testing.assert_close(_predict(restored), trained)

    undated = _tiny_model()
    prepare_for_finetuning(undated, FinetuneConfig(date=False, lora_rank=2))
    assert undated.date is None and trainable_state_dict(undated)["date"] == {}
    with pytest.raises(ValueError, match="no date conditioner"):
        load_trainable_state_dict(undated, state)
    no_lora = _tiny_model()
    prepare_for_finetuning(no_lora, FinetuneConfig())
    with pytest.raises(KeyError, match="LoRA state mismatch"):
        load_trainable_state_dict(no_lora, state)


def test_finetune_config_validation():
    cfg = FinetuneConfig(lora_rank=8, lora_targets=["attn1.to_q"])
    assert cfg.lora_targets == ("attn1.to_q",) and FinetuneConfig(**asdict(cfg)) == cfg
    assert FinetuneConfig().lora_targets == DEFAULT_LORA_TARGETS and FinetuneConfig().lora_rank == 0
    for bad in ({"date_freqs": 0}, {"date_hidden": 0}, {"lora_rank": -1}, {"lora_alpha": 0.0},
                {"lora_rank": 4, "lora_targets": ()}, {"lora_targets": "attn1.to_q"}):
        with pytest.raises(ValueError):
            FinetuneConfig(**bad)


def test_model_without_conditioner_and_sampling_pipeline(tmp_path):
    model = _tiny_model()
    with torch.no_grad():
        base = _predict(model)
        with model.dates(YEARS[:3]):                                            # no conditioner: dates are ignored
            assert torch.equal(_predict(model), base)
    assert model.date_stats() is None
    assert model.device == torch.device("cpu")

    (tmp_path / "scheduler").mkdir()                                            # what SD 2.1 ships; MVGenMaster's
    (tmp_path / "scheduler" / "scheduler_config.json").write_text(json.dumps({  # config then changes the betas
        "_class_name": "DDIMScheduler", "beta_start": 0.00085, "beta_end": 0.012, "beta_schedule": "scaled_linear",
        "clip_sample": False, "num_train_timesteps": 1000, "prediction_type": "v_prediction",
        "set_alpha_to_one": False, "steps_offset": 1}))
    model.config = EasyDict(pretrained_model_name_or_path=str(tmp_path), rescale_betas_zero_snr=True,
                            beta_schedule="snr_rescale", snr_rescale=4)
    pipeline = model.sampling_pipeline()
    assert pipeline.unet is model.unet and pipeline.vae is model.vae
    assert type(pipeline.scheduler).__name__ == "DDIMScheduler"
    assert pipeline.scheduler.config.prediction_type == "v_prediction"


def test_tf32_matmul_is_refused():
    """TF32 would round the fp32 matrix products of MVGenMaster's prior warp; both of PyTorch's switches are caught."""
    assert_exact_matmul()                                                       # PyTorch's default: off
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        with pytest.raises(AssertionError, match="TF32"):
            assert_exact_matmul()
        with pytest.raises(AssertionError, match="TF32"):
            load_mvgen("no/such/model", device="cpu")                           # refused before anything is read
        torch.backends.cuda.matmul.allow_tf32 = False
        assert_exact_matmul()
        torch.set_float32_matmul_precision("high")
        with pytest.raises(AssertionError, match="TF32"):
            assert_exact_matmul()
    finally:
        torch.set_float32_matmul_precision("highest")
    assert_exact_matmul()
    with pytest.raises(FileNotFoundError):
        load_mvgen("no/such/model", device="cpu")                               # past the check


# ----------------------------------------------------------------------------- LoRA on the layers that read the date

def _attach_by_hand(model: MVGen, lora_first: bool) -> None:
    """The date conditioner and LoRA with ``time_emb_proj`` among its targets, attached in the given order."""
    def attach_lora() -> None:
        add_lora(model.unet, READER_LORA_TARGETS, rank=2)

    def attach_date() -> None:
        model.date = DateConditioner(model.unet, DateEmbedding(temb_width(model.unet)))

    for attach in ((attach_lora, attach_date) if lora_first else (attach_date, attach_lora)):
        attach()


def _reader_adapters(model: MVGen) -> list[LoRALinear]:
    """The LoRA layers around the ResBlocks' ``time_emb_proj``."""
    return [layer for name, layer in lora_layers(model.unet).items() if name.endswith("." + TIME_PROJ_TARGET)]


def test_time_proj_target_is_optional_and_names_every_reader_of_the_date():
    assert TIME_PROJ_TARGET == "time_emb_proj"
    assert TIME_PROJ_TARGET not in DEFAULT_LORA_TARGETS and FinetuneConfig().lora_targets == DEFAULT_LORA_TARGETS
    unet = _tiny_model().unet
    blocks = dated_blocks(unet)
    wrapped = add_lora(unet, (TIME_PROJ_TARGET,), rank=2)
    assert wrapped == [f"{name}.{TIME_PROJ_TARGET}" for name, m in unet.named_modules()
                       if isinstance(m, ResnetBlock2D)]
    assert len(wrapped) == 8 and all(isinstance(block.time_emb_proj, LoRALinear) for block in blocks)
    assert dated_blocks(unet) == blocks and temb_width(unet) == 128             # both see through the wrapper


def test_date_and_reader_lora_attach_in_either_order():
    """An exact no-op at initialisation in both orders, and one and the same model once they hold equal weights."""
    years, valid = YEARS[:3], torch.tensor([True, True, False])
    models = {}
    for lora_first in (False, True):
        model = _tiny_model()
        with torch.no_grad():
            base = _predict(model, batch=2)
        _attach_by_hand(model, lora_first)
        assert len(lora_layers(model.unet)) == 24 + 8 and len(_reader_adapters(model)) == 8
        with torch.no_grad(), model.dates(years, valid, uncond="unknown"):
            assert torch.equal(_predict(model, batch=2), base)                  # step 0: the base model, bit for bit
        assert model.date_stats()["date_rms"] == 0.0
        models[lora_first] = model

    _train_adapters(models[False])
    load_trainable_state_dict(models[True], trainable_state_dict(models[False]))
    outputs = {}
    for lora_first, model in models.items():
        with torch.no_grad(), model.dates(years, valid, uncond="unknown"):
            outputs[lora_first] = _predict(model, batch=2)
    assert not torch.allclose(outputs[False], base) and torch.equal(outputs[True], outputs[False])


def _reader_gradients(checkpointing: bool, trained: bool) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Per reader, the gradients of its LoRA's ``down`` and ``up`` matrices through MVGenMaster's forward pass."""
    model = _tiny_model()
    prepare_for_finetuning(model, FinetuneConfig(lora_rank=2, lora_targets=READER_LORA_TARGETS))
    if trained:
        _train_adapters(model)
    model.unet.train(checkpointing)                 # MVGenMaster's blocks only checkpoint in training mode
    with model.dates(YEARS[:3], torch.tensor([True, True, False])):
        out = _predict(model)
    out.pow(2).mean().backward()
    readers = _reader_adapters(model)
    assert len(readers) == 8
    return [layer.down.weight.grad for layer in readers], [layer.up.weight.grad for layer in readers]


def test_gradients_reach_the_lora_of_every_reader():
    down, up = _reader_gradients(checkpointing=True, trained=False)
    assert all(torch.count_nonzero(grad) == 0 for grad in down)         # step 0: ``up`` is zero, so only it can move
    assert all(grad.abs().sum() > 0 for grad in up)
    down, up = _reader_gradients(checkpointing=True, trained=True)
    assert all(grad.abs().sum() > 0 for grad in down + up)
    plain_down, plain_up = _reader_gradients(checkpointing=False, trained=True)
    for checkpointed, plain in zip(down + up, plain_down + plain_up, strict=True):
        torch.testing.assert_close(checkpointed, plain, rtol=1e-5, atol=1e-8)


def test_reader_lora_under_bf16_is_the_base_model_up_to_rounding():
    """``temb`` is fp32 under bf16 autocast, so a wrapped ``time_emb_proj`` returns fp32 where the layer returned bf16.

    Step 0 of a bf16 run then differs from the released model by rounding, on a CPU as well; its own dated and
    undated passes still agree bit for bit, which is what a step-0 probe compares.
    """
    model = _tiny_model()
    with torch.no_grad():
        base = _predict(model)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            base_bf16 = _predict(model)
    prepare_for_finetuning(model, FinetuneConfig(lora_rank=2, lora_targets=(TIME_PROJ_TARGET,)))
    dtypes = {}
    reader = _reader_adapters(model)[0]
    reader.register_forward_hook(lambda module, args, output: dtypes.update(wrapped=output.dtype, temb=args[0].dtype))
    reader.base.register_forward_hook(lambda module, args, output: dtypes.update(base=output.dtype))
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        undated = _predict(model)
        with model.dates(YEARS[:3]):
            dated = _predict(model)
    assert dtypes == {"temb": torch.float32, "base": torch.bfloat16, "wrapped": torch.float32}
    assert torch.equal(dated, undated)
    rounding = (base_bf16.float() - base).abs().max()
    assert 0 < (undated.float() - base_bf16.float()).abs().max() < 2 * rounding
    assert (undated.float() - base).abs().max() < 2 * rounding


def test_reader_lora_round_trips_through_a_checkpoint(tmp_path, monkeypatch):
    cfg = FinetuneConfig(lora_rank=2, lora_targets=list(READER_LORA_TARGETS))
    model = _tiny_model()
    prepare_for_finetuning(model, cfg)
    _train_adapters(model)
    years, valid = YEARS[:3], torch.tensor([True, False, True])
    with torch.no_grad(), model.dates(years, valid):
        trained = _predict(model)
    state = trainable_state_dict(model)["lora"]
    reader_keys = [key for key in state if key.endswith((f"{TIME_PROJ_TARGET}.down", f"{TIME_PROJ_TARGET}.up"))]
    assert len(state) == 2 * (24 + 8) and len(reader_keys) == 2 * 8

    save_checkpoint(tmp_path / "step_3.pt", model, cfg, step=3)
    monkeypatch.setattr(mvgen_model, "load_mvgen", lambda model_dir, device: _tiny_model())
    restored, ckpt = load_finetuned(tmp_path / "step_3.pt", device="cpu")
    assert FinetuneConfig(**ckpt["finetune_config"]) == cfg
    assert TIME_PROJ_TARGET in ckpt["finetune_config"]["lora_targets"] and len(_reader_adapters(restored)) == 8
    with torch.no_grad(), restored.dates(years, valid):
        torch.testing.assert_close(_predict(restored), trained)
