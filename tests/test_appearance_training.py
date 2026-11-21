"""Wiring tests for the appearance pathway through the model, config and loss.

The experiment this supports is "date FiLM alone" versus "date FiLM + appearance
FiLM", so what has to hold is that appearance is a *separate* pathway: it must
reach the network, be trainable and checkpointable on its own, and leave the
date path exactly as it was. The tests that need SEVA's real ResBlock skip
without it, as elsewhere in this suite; the rest run anywhere.
"""
import pytest
import torch
from torch import nn

# Only modules that do NOT pull in SEVA may be imported above the skip below:
# finetune imports dated_seva, which imports seva, so it has to come after.
from timemachine.model.appearance_film import AppearanceEncoder, AppearanceFiLMConv, AppearanceState
from timemachine.training.loss import LossConfig, draw_appearance_validity

seva = pytest.importorskip("seva.model", reason="SEVA not installed (scripts/setup_seva.sh)")

from timemachine.model.dated_seva import DatedSGMWrapper               # noqa: E402
from timemachine.model.finetune import (                               # noqa: E402
    APPEARANCE_INJECTIONS, FinetuneConfig, load_trainable_state_dict, prepare_for_finetuning,
    trainable_state_dict,
)
from timemachine.training.loss import training_loss                    # noqa: E402

FEATURES = 14


def tiny_model(appearance="film", date="film", lora_rank=0):
    model = DatedSGMWrapper(seva.Seva(model_channels=8, num_res_blocks=2))
    cfg = FinetuneConfig(lora_rank=lora_rank, date_injection=date, appearance_injection=appearance,
                         film_dim=32, appearance_dim=32, grad_checkpointing=False)
    return model, cfg, prepare_for_finetuning(model, cfg)


def forward_args(n_views=4):
    return (torch.randn(n_views, 4, 8, 8), torch.arange(n_views),
            {"crossattn": None, "dense_vector": None})


# ------------------------------------------------------------------ config
def test_config_validates_the_injection_choice():
    assert FinetuneConfig(appearance_injection="none").appearance_injection == "none"
    assert set(APPEARANCE_INJECTIONS) == {"film", "none"}
    with pytest.raises(ValueError, match="appearance_injection"):
        FinetuneConfig(appearance_injection="sepia")


def test_appearance_dropout_drops_whole_samples_together():
    """A half-described sample would let the model infer the missing half from the rest."""
    device = torch.device("cpu")
    always = draw_appearance_validity(6, LossConfig(appearance_dropout=0.0), device)
    never = draw_appearance_validity(6, LossConfig(appearance_dropout=1.0), device)
    assert always.all() and not never.any()
    assert always.shape == (6,) and always.dtype == torch.bool

    seen = set()
    for i in range(60):
        g = torch.Generator().manual_seed(i)
        v = draw_appearance_validity(5, LossConfig(appearance_dropout=0.5), device, g)
        assert len(set(v.tolist())) == 1          # all views of a sample agree
        seen.add(bool(v[0]))
    assert seen == {True, False}                  # and both outcomes do happen


# ----------------------------------------------------------- parameter groups
def test_appearance_adds_its_own_trainable_group_and_freezes_seva():
    model, _, groups = tiny_model()
    assert groups["appearance"] and all(p.requires_grad for p in groups["appearance"])
    assert not any(p.requires_grad for p in model.module.head.parameters())
    assert groups["film"] and groups["appearance"]           # both pathways, separately
    assert set(groups) == {"date", "film", "lora", "appearance"}


def test_no_appearance_group_when_the_pathway_is_off():
    _, _, groups = tiny_model(appearance="none")
    assert groups["appearance"] == []


def test_appearance_works_without_any_date_pathway():
    """The control arm of the experiment: appearance only, dates off."""
    model, _, groups = tiny_model(appearance="film", date="none")
    assert groups["appearance"] and not groups["film"] and not groups["date"]
    x, t, c = forward_args()
    before = model(x, t, c, num_frames=4)
    for p in model.appearance_parameters():
        nn.init.normal_(p, std=0.05)
    after = model(x, t, c, appearance=torch.randn(4, FEATURES), num_frames=4)
    assert not torch.allclose(before, after)


# --------------------------------------------------------------- the model
def test_zero_init_leaves_seva_untouched_with_both_conditionings():
    model, _, _ = tiny_model()
    x, t, c = forward_args()
    plain = model(x, t, c, num_frames=4)
    conditioned = model(x, t, c, date=torch.tensor([1880., 1880., 2015., 2015.]),
                        appearance=torch.randn(4, FEATURES), num_frames=4)
    torch.testing.assert_close(plain, conditioned)


def test_date_and_appearance_move_the_output_independently():
    torch.manual_seed(0)
    model, _, groups = tiny_model()
    for p in groups["film"] + groups["appearance"]:
        nn.init.normal_(p, std=0.05)
    x, t, c = forward_args()
    old, modern = torch.full((4,), 1880.), torch.full((4,), 2015.)
    look_a, look_b = torch.randn(4, FEATURES), torch.randn(4, FEATURES)
    run = lambda d, a: model(x, t, c, date=d, appearance=a, num_frames=4)

    assert not torch.allclose(run(old, look_a), run(modern, look_a))   # date alone moves it
    assert not torch.allclose(run(old, look_a), run(old, look_b))      # appearance alone moves it
    # Both can be held while the other varies, which is what "factorized" means here.
    assert not torch.allclose(run(old, look_a), run(modern, look_b))


def test_raw_conditioning_inputs_are_cleared_but_encoded_features_survive_the_forward():
    """Both halves matter, and they pull in opposite directions.

    The raw inputs must be cleared so they cannot leak into an unrelated batch. The
    *encoded* features must NOT be, because gradient checkpointing recomputes the
    blocks during backward - after the context has exited - and they have to see the
    same tensor the forward used. Appearance follows the contract date FiLM already
    set; clearing them here would silently drop the conditioning in the backward pass.
    """
    model, _, _ = tiny_model()
    x, t, c = forward_args()
    model(x, t, c, date=torch.full((4,), 1900.), appearance=torch.randn(4, FEATURES), num_frames=4)
    hook = model.date_module
    assert hook.appearance is None and hook.appearance_valid is None
    assert hook.year is None and hook.valid is None
    assert hook.appearance_state.features is not None     # still available for the backward
    assert hook.state.features is not None                # ... exactly as the date path does

    # A later forward with no conditioning clears them, so nothing leaks into it.
    model(x, t, c, num_frames=4)
    assert hook.appearance_state.features is None and hook.state.features is None


def test_classifier_free_guidance_doubles_the_batch_safely():
    """SEVA's sampler stacks [unconditional, conditional]; appearance is repeated to both."""
    model, _, groups = tiny_model()
    for p in groups["appearance"]:
        nn.init.normal_(p, std=0.05)
    stats = torch.randn(4, FEATURES)
    doubled = model(torch.randn(8, 4, 8, 8), torch.arange(8),
                    {"crossattn": None, "dense_vector": None}, appearance=stats, num_frames=4)
    assert doubled.shape == (8, 4, 8, 8)

    with pytest.raises(ValueError, match="appearance rows"):
        model(torch.randn(6, 4, 8, 8), torch.arange(6),
              {"crossattn": None, "dense_vector": None}, appearance=stats, num_frames=4)


def test_unknown_appearance_differs_from_a_given_one():
    model, _, groups = tiny_model()
    for p in groups["appearance"]:
        nn.init.normal_(p, std=0.05)
    x, t, c = forward_args()
    stats = torch.randn(4, FEATURES)
    known = model(x, t, c, appearance=stats, appearance_valid=torch.ones(4, dtype=torch.bool), num_frames=4)
    unknown = model(x, t, c, appearance=stats, appearance_valid=torch.zeros(4, dtype=torch.bool), num_frames=4)
    assert not torch.allclose(known, unknown)


# ------------------------------------------------------------- checkpoints
def test_appearance_round_trips_through_a_checkpoint():
    model, cfg, _ = tiny_model()
    model.date_module.appearance_encoder.fit_normalization(torch.randn(50, FEATURES))
    for p in model.appearance_parameters():
        nn.init.normal_(p, std=0.1)

    state = trainable_state_dict(model)
    assert "appearance" in state and set(state["appearance"]) == {"encoder", "layers"}

    restored, _, _ = tiny_model()
    load_trainable_state_dict(restored, state)
    for a, b in zip(model.appearance_parameters(), restored.appearance_parameters()):
        assert torch.equal(a, b)
    # The standardization buffers must travel too, or a reloaded model sees different inputs.
    assert torch.equal(model.date_module.appearance_encoder.mean,
                       restored.date_module.appearance_encoder.mean)
    assert torch.equal(model.date_module.appearance_encoder.scale,
                       restored.date_module.appearance_encoder.scale)


def test_a_date_only_checkpoint_still_loads(tmp_path):
    """Older runs have no appearance section; loading one must not fail."""
    model, _, _ = tiny_model(appearance="none")
    state = trainable_state_dict(model)
    assert "appearance" not in state
    load_trainable_state_dict(model, state)


# -------------------------------------------------------------- the loss
def _loss_inputs(n_views=4):
    latents = torch.randn(n_views, 4, 8, 8)
    replace = torch.cat([torch.randn(n_views, 4, 8, 8), torch.zeros(n_views, 1, 8, 8)], 1)
    replace[:2, -1] = 1.0                                   # first two views are inputs
    cond = {"crossattn": None, "dense_vector": None, "replace": replace}
    interval = torch.tensor([[1880., 1881.]] * 2 + [[2015., 2016.]] * 2)
    is_input = torch.tensor([True, True, False, False])
    return latents, cond, interval, is_input


def test_training_loss_reaches_the_appearance_parameters():
    torch.manual_seed(0)
    model, _, groups = tiny_model()
    latents, cond, interval, is_input = _loss_inputs()
    loss, info = training_loss(model, torch.linspace(0.002, 80, 1000), latents, cond, dict(cond),
                               interval, LossConfig(appearance_dropout=0.0),
                               torch.Generator().manual_seed(0), is_input=is_input,
                               appearance=torch.randn(4, FEATURES))
    assert torch.isfinite(loss) and info["appearance_known"] is True
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in groups["appearance"])


def test_training_loss_without_appearance_reports_none_and_still_works():
    model, _, _ = tiny_model(appearance="none")
    latents, cond, interval, is_input = _loss_inputs()
    loss, info = training_loss(model, torch.linspace(0.002, 80, 1000), latents, cond, dict(cond),
                               interval, LossConfig(), torch.Generator().manual_seed(0), is_input=is_input)
    assert torch.isfinite(loss) and info["appearance_known"] is None
