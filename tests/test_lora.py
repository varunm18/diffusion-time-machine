"""LoRA adapters and gradient checkpointing (CPU, on small stand-in modules)."""
import torch
from torch import nn

from timemachine.model.grad_checkpoint import enable_gradient_checkpointing
from timemachine.model.lora import (
    LoRALinear, add_lora, load_lora_state_dict, lora_parameters, lora_state_dict,
)


class Attn(nn.Module):
    def __init__(self, d=8):
        super().__init__()
        self.to_q, self.to_k = nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False)
        self.to_out = nn.Sequential(nn.Linear(d, d))

    def forward(self, x):
        return self.to_out(self.to_q(x) * self.to_k(x).sigmoid())


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn1, self.other = Attn(), nn.Linear(8, 8)

    def forward(self, x):
        return self.other(self.attn1(x)) + x


def test_lora_is_identity_at_init_and_trains_only_adapters():
    torch.manual_seed(0)
    model = nn.Sequential(Block(), Block())
    x = torch.randn(4, 8)
    before = model(x)
    names = add_lora(model, ("attn1.to_q", "attn1.to_out.0"), rank=2)
    assert names == ["0.attn1.to_q", "0.attn1.to_out.0", "1.attn1.to_q", "1.attn1.to_out.0"]
    assert isinstance(model[0].attn1.to_out[0], LoRALinear) and not isinstance(model[0].attn1.to_k, LoRALinear)
    torch.testing.assert_close(model(x), before)
    model.requires_grad_(False)
    for p in lora_parameters(model):
        p.requires_grad_(True)
    model(x).sum().backward()
    assert model[0].attn1.to_q.base.weight.grad is None
    assert model[0].attn1.to_q.up.weight.grad.abs().sum() > 0


def test_lora_state_dict_round_trip():
    torch.manual_seed(0)
    a, b = nn.Sequential(Block()), nn.Sequential(Block())
    b.load_state_dict(a.state_dict())
    for m in (a, b):
        add_lora(m, ("to_q",), rank=3)
    nn.init.normal_(a[0].attn1.to_q.up.weight)
    x = torch.randn(2, 8)
    assert not torch.allclose(a(x), b(x))
    load_lora_state_dict(b, lora_state_dict(a))
    torch.testing.assert_close(a(x), b(x))
    assert set(lora_state_dict(a)) == {"0.attn1.to_q.down", "0.attn1.to_q.up"}


def test_lora_keeps_an_fp32_sum_for_fp32_input_under_bf16_autocast():
    """LoRALinear's docstring: the wrapped layer returns fp32 there, so that a small update is not rounded away."""
    torch.manual_seed(0)
    base = nn.Linear(8, 8)
    layer = LoRALinear(base, rank=2)
    x = torch.randn(16, 8)
    assert torch.equal(layer(x), base(x))                               # outside autocast: bit for bit
    nn.init.normal_(layer.up.weight, std=1e-5)                          # an update far below bf16's resolution
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        plain, adapted = base(x), layer(x)
        assert plain.dtype == torch.bfloat16 and adapted.dtype == torch.float32
        assert layer(x.bfloat16()).dtype == torch.bfloat16
        large = plain.float().abs() > 0.1                               # bf16 resolves these to 2e-4 at best
        update = (adapted - plain.float())[large]
        assert torch.count_nonzero(update) > 0.9 * update.numel() > 0   # the fp32 sum keeps the update
        assert torch.equal(plain[large] + update.bfloat16(), plain[large])      # a bf16 sum would lose it


class TinyUNet(nn.Module):
    """Has Seva's block attributes and calling convention (keyword args)."""

    def __init__(self):
        super().__init__()
        self.input_blocks = nn.ModuleList([nn.Linear(8, 8), nn.Linear(8, 8)])
        self.middle_block = nn.Linear(8, 8)
        self.output_blocks = nn.ModuleList([nn.Linear(8, 8)])

    def forward(self, x):
        for m in [*self.input_blocks, self.middle_block, *self.output_blocks]:
            x = torch.tanh(m(x))
        return x


def test_gradient_checkpointing_keeps_outputs_grads_and_keys():
    torch.manual_seed(0)
    plain, ckpt = TinyUNet(), TinyUNet()
    ckpt.load_state_dict(plain.state_dict())
    assert enable_gradient_checkpointing(ckpt) == 4
    assert enable_gradient_checkpointing(ckpt) == 4          # idempotent
    assert list(ckpt.state_dict()) == list(plain.state_dict())
    x = torch.randn(3, 8, requires_grad=True)
    for m in (plain, ckpt):
        m(x).pow(2).sum().backward()
    for p, q in zip(plain.parameters(), ckpt.parameters()):
        torch.testing.assert_close(p.grad, q.grad)
    with torch.no_grad():
        torch.testing.assert_close(plain(x), ckpt(x))


def test_diagnostics_on_fresh_and_perturbed_modules():
    from timemachine.model.date_embedding import DateEmbedding
    from timemachine.model.diagnostics import date_embedding_rms, layer_kind, lora_relative_update, summarize_lora

    date = DateEmbedding(out_dim=16)
    assert all(v == 0 for v in date_embedding_rms(date).values())          # zero-init
    nn.init.normal_(date.mlp[-1].weight)
    assert date_embedding_rms(date)["date_rms_spread"] > 0
    base, down, up = torch.randn(8, 4), torch.randn(2, 4), torch.zeros(8, 2)
    assert lora_relative_update(down, up, 1.0, base) == 0
    assert layer_kind("input_blocks.4.1.time_mix_blocks.0.ff_in.net.2") == "timemix.ff_in"
    assert layer_kind("middle_block.1.transformer_blocks.0.attn2.to_q") == "spatial.attn2"
    summary = summarize_lora({"a.transformer_blocks.0.attn1.to_q": 0.1, "b.transformer_blocks.0.attn1.to_k": 0.3})
    assert summary["lora_rel_spatial.attn1_max"] == 0.3 and summary["lora_rel_max"] == 0.3
