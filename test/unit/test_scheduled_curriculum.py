"""
Unit tests for the scheduled curriculum mode of OnlineTargetForecaster.

The forcing bound anneals from show_trg_max_idx_start to show_trg_max_idx_end
over [epoch_show_trg, show_trg_schedule_end_frac * max_epochs]. Tests use a
bare instance (via __new__) to avoid building the full ProT model.
"""

from unittest.mock import PropertyMock, patch

import pytest

from proT.training.forecasters.online_target_forecaster import OnlineTargetForecaster


def _make_bare_forecaster(
    epoch_show_trg=10,
    max_epochs=100,
    start=150,
    end=0,
    schedule="linear",
    end_frac=1.0,
    random_draw=False,
):
    """Bare OnlineTargetForecaster instance with schedule attributes set."""
    f = OnlineTargetForecaster.__new__(OnlineTargetForecaster)
    f.epoch_show_trg = epoch_show_trg
    f.max_epochs = max_epochs
    f.show_trg_max_idx_start = start
    f.show_trg_max_idx_end = end
    f.show_trg_schedule = schedule
    f.show_trg_schedule_end_frac = end_frac
    f.show_trg_random_draw = random_draw
    f.show_trg_max_idx = None
    f.show_trg_active = False
    return f


class TestScheduledBound:
    """Tests for _scheduled_bound (deterministic schedule math)."""

    def test_linear_boundaries(self):
        f = _make_bare_forecaster(epoch_show_trg=10, max_epochs=100, start=150, end=0)
        assert f._scheduled_bound(0) == 150    # before curriculum start
        assert f._scheduled_bound(10) == 150   # at curriculum start
        assert f._scheduled_bound(99) == 0     # last epoch
        assert f._scheduled_bound(150) == 0    # beyond max_epochs (clamped)

    def test_linear_midpoint(self):
        f = _make_bare_forecaster(epoch_show_trg=10, max_epochs=100, start=150, end=0)
        # end_ep = 99; midpoint frac = (55-10)/(99-10) ~ 0.506 -> bound ~ 74
        assert f._scheduled_bound(55) == pytest.approx(74, abs=2)

    def test_linear_monotonic_decrease(self):
        f = _make_bare_forecaster(epoch_show_trg=10, max_epochs=100, start=150, end=0)
        bounds = [f._scheduled_bound(e) for e in range(10, 100)]
        assert all(b1 >= b2 for b1, b2 in zip(bounds, bounds[1:]))

    def test_cosine_midpoint_higher_than_linear(self):
        f_cos = _make_bare_forecaster(schedule="cosine", epoch_show_trg=10, max_epochs=100)
        f_lin = _make_bare_forecaster(schedule="linear", epoch_show_trg=10, max_epochs=100)
        # cosine annealing decreases slowly at first -> higher bound early on
        # epoch 32 ~ frac 0.25: cosine ~128, linear ~112
        assert f_cos._scheduled_bound(32) == pytest.approx(128, abs=2)
        assert f_lin._scheduled_bound(32) == pytest.approx(112, abs=2)
        assert f_cos._scheduled_bound(32) > f_lin._scheduled_bound(32)
        # same endpoints
        assert f_cos._scheduled_bound(10) == f_lin._scheduled_bound(10) == 150
        assert f_cos._scheduled_bound(99) == f_lin._scheduled_bound(99) == 0

    def test_end_frac_reaches_end_early(self):
        f = _make_bare_forecaster(epoch_show_trg=10, max_epochs=100, start=150, end=0,
                                  end_frac=0.8)
        # end_ep = 80
        assert f._scheduled_bound(80) == 0
        assert f._scheduled_bound(90) == 0
        assert f._scheduled_bound(79) > 0

    def test_nonzero_end(self):
        f = _make_bare_forecaster(epoch_show_trg=10, max_epochs=100, start=150, end=20)
        assert f._scheduled_bound(99) == 20
        assert f._scheduled_bound(10) == 150


class TestUpdateScheduledBound:
    """Tests for _update_scheduled_bound (random draw vs deterministic)."""

    @pytest.mark.parametrize("epoch,expected", [(10, 150), (55, 74), (99, 0)])
    def test_deterministic_draw(self, epoch, expected):
        f = _make_bare_forecaster(epoch_show_trg=10, max_epochs=100, start=150, end=0,
                                  random_draw=False)
        with patch.object(OnlineTargetForecaster, "current_epoch",
                          new_callable=PropertyMock, return_value=epoch):
            f._update_scheduled_bound()
        assert f.show_trg_max_idx == pytest.approx(expected, abs=2)

    def test_random_draw_within_bound(self):
        f = _make_bare_forecaster(epoch_show_trg=10, max_epochs=100, start=150, end=0,
                                  random_draw=True)
        with patch.object(OnlineTargetForecaster, "current_epoch",
                          new_callable=PropertyMock, return_value=10):
            for _ in range(20):
                f._update_scheduled_bound()
                assert 0 <= f.show_trg_max_idx <= 150

    def test_random_draw_at_zero_bound(self):
        f = _make_bare_forecaster(epoch_show_trg=10, max_epochs=100, start=150, end=0,
                                  random_draw=True)
        with patch.object(OnlineTargetForecaster, "current_epoch",
                          new_callable=PropertyMock, return_value=99):
            f._update_scheduled_bound()
        assert f.show_trg_max_idx == 0
