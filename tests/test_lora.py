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
