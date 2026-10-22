"""Unit tests for the standalone teacher-forcing curriculum."""
import math
import random

import pytest

from proT.training.curriculum import ForcingCurriculum


def sched(**kw):
    base = dict(mode="scheduled", units="fraction", epoch_show_trg=0,
                max_epochs=2000, start=0.8, end=0.0, schedule="linear",
                schedule_end_frac=0.8, random_draw=False)
    base.update(kw)
    return ForcingCurriculum(**base)


def test_random_per_sample_draw():
    import torch
    c = ForcingCurriculum(mode="random_per_sample", units="fraction",
                          max_epochs=10, upper_bound_max=0.9, p_zero=0.3)
    f = c.draw_per_sample(20000, generator=torch.Generator().manual_seed(0))
    assert f.shape == (20000,)
    assert (f >= 0).all() and (f <= 0.9).all()
    assert abs((f == 0).float().mean().item() - 0.3) < 0.02
    with pytest.raises(AssertionError):
        ForcingCurriculum(mode="random_per_sample", units="index",
                          max_epochs=10, upper_bound_max=5)


def test_no_warmup_regression():
    # warmup None/0: hold start until onset, linear anneal to end at 1600
    for warm in (None, 0):
        c = sched(warmup_epoch=warm)
        assert c.scheduled_bound(0) == 0.8
        assert c.scheduled_bound(400) == pytest.approx(0.8 - 0.8 * 400 / 1600)
        assert c.scheduled_bound(1600) == 0.0
        assert c.scheduled_bound(1999) == 0.0


def test_warmup_full_forcing():
    c = sched(warmup_epoch=50)
    for ep in (0, 25, 49):
        assert c.scheduled_bound(ep) == 1.0       # full sequence given
    assert c.scheduled_bound(50) == 0.8           # anneal starts after warmup
    # anneal runs over [50, 1600]: midpoint at (50+1600)/2
    assert c.scheduled_bound(825) == pytest.approx(0.8 - 0.8 * (825 - 50) / 1550)
    assert c.scheduled_bound(1600) == 0.0


def test_cosine_and_ramp():
    c = sched(schedule="cosine")
    assert c.scheduled_bound(0) == 0.8
    assert c.scheduled_bound(800) == pytest.approx(0.4)  # cosine midpoint
    # ramp-up phase (legacy mechanism, still supported after warmup):
    # ramp occupies the first 10% of [50, 1600] -> 155 epochs
    c2 = sched(warmup_epoch=50, ramp_up_frac=0.1, ramp_start=1.0)
    assert c2.scheduled_bound(50) == 1.0                        # ramp start
    assert c2.scheduled_bound(50 + 77) == pytest.approx(1.0 - 0.2 * 77 / 155, abs=1e-3)


def test_random_draw_within_bound():
    c = sched(random_draw=True)
    random.seed(0)
    for ep in (0, 400, 800):
        d = c.draw(ep)
        assert 0.0 <= d <= c.scheduled_bound(ep)


def test_index_units_cast_to_int():
    c = ForcingCurriculum(mode="scheduled", units="index", epoch_show_trg=10,
                          max_epochs=100, start=80, end=0, random_draw=False,
                          warmup_epoch=5)
    assert c.scheduled_bound(12) == 10 ** 9      # warmup: full forcing
    assert c.scheduled_bound(15) == 80
    assert isinstance(c.scheduled_bound(50), int)


def test_warmup_validation():
    with pytest.raises(AssertionError):
        sched(warmup_epoch=-1)
    with pytest.raises(AssertionError):
        sched(warmup_epoch=1700)  # warmup must end before the schedule end
    with pytest.raises(AssertionError):
        sched(warmup_epoch=0, warmup_bound=0.9)  # bound requires warmup_epoch > 0
    with pytest.raises(AssertionError):
        sched(warmup_epoch=50, warmup_bound=1.5)  # fraction bounds in [0, 1]


def test_warmup_bound_constant_level():
    # warmup_bound: constant 90% forcing during warmup, then anneal from start
    c = sched(warmup_epoch=50, warmup_bound=0.9)
    for ep in (0, 25, 49):
        assert c.scheduled_bound(ep) == 0.9
    assert c.scheduled_bound(50) == 0.8  # anneal starts at start (0.8)
    # warmup_bound=None reproduces full forcing
    assert sched(warmup_epoch=50).scheduled_bound(25) == 1.0


def test_from_config_keys():
    cfg = {"training": {"max_epochs": 2000, "target_show_mode": "scheduled",
                        "show_trg_units": "fraction", "show_trg_max_idx_start": 0.8,
                        "warmup_epoch": 50}}
    c = ForcingCurriculum.from_config(cfg)
    assert c.warmup_epoch == 50 and c.start == 0.8 and c.units == "fraction"
