"""
Teacher-forcing curriculum for the GRU decoder heads ("show target" schedule).

All forcing settings and schedule logic live here, in one unit-testable place;
the forecaster only calls `ForcingCurriculum.draw(epoch)` once per epoch.

Schedule (mode="scheduled"), forcing upper bound vs epoch:

    epoch < epoch_show_trg                : inactive (bound unused)
    [epoch_show_trg, +warmup_epoch)       : WARMUP - constant forcing level
                                            (warmup_bound; default full forcing),
                                            skipped when warmup_epoch is 0/None
    [warmup_end, +ramp_up)                : optional ramp ramp_start -> start
    afterwards                            : anneal start -> end (linear/cosine),
                                            end reached at schedule_end_frac * max_epochs

Units: "fraction" (bounds are fractions of each sample's own valid target
length) or "index" (absolute position index). An optional per-epoch uniform
random draw in [draw_lower, min(draw_upper, bound)] (default [0, bound])
decorrelates the forcing point from the epoch (random_draw, default True).
"""

import math
import random


class ForcingCurriculum:
    def __init__(
        self,
        mode=None,                    # None | "deterministic" | "random" | "scheduled"
        #   | "random_per_sample" (one fraction per sample, every step)
        units="index",                # "index" | "fraction"
        epoch_show_trg=0,             # curriculum onset epoch
        max_epochs=100,
        show_trg_max_idx=None,        # fixed bound (deterministic mode)
        upper_bound_max=None,         # draw upper bound (random mode)
        start=None,                   # scheduled mode: initial bound
        end=0,                        # scheduled mode: final bound
        schedule="linear",            # "linear" | "cosine"
        schedule_end_frac=1.0,        # end reached at this fraction of max_epochs
        random_draw=True,             # per-epoch uniform draw in [0, bound]
        draw_lower=0.0,               # random draw window: lower end (start value;
        #   anneals to draw_lower_end on the same schedule as the bound)
        draw_lower_end=0.0,           # lower end at the end of the schedule
        draw_upper=None,              # upper end (None = the current schedule bound)
        warmup_epoch=None,            # warmup length; 0/None = skip
        warmup_bound=None,            # warmup forcing level (fraction of each
        #   sample's own valid length / absolute index); None = full forcing
        ramp_up_frac=0.0,             # optional ramp ramp_start -> start
        ramp_start=None,
        p_zero=0.0,                   # random_per_sample: probability of an
        #   exact zero fraction (pure forecast, nothing shown)
    ):
        assert units in ("index", "fraction"), \
            f"show_trg_units must be 'index' or 'fraction', got {units}"
        if units == "fraction":
            for name, val in (("show_trg_max_idx", show_trg_max_idx),
                              ("show_trg_upper_bound_max", upper_bound_max),
                              ("show_trg_max_idx_start", start),
                              ("show_trg_max_idx_end", end),
                              ("show_trg_ramp_start", ramp_start),
                              ("show_trg_draw_lower", draw_lower),
                              ("show_trg_draw_lower_end", draw_lower_end),
                              ("show_trg_draw_upper", draw_upper)):
                if val is not None:
                    assert 0.0 <= val <= 1.0, \
                        f"{name} must be in [0, 1] when show_trg_units='fraction', got {val}"
        if mode == "random":
            assert upper_bound_max is not None, \
                "show_trg_upper_bound_max must be set for random target mode"
        if mode == "random_per_sample":
            assert units == "fraction", \
                "random_per_sample requires show_trg_units='fraction'"
            assert upper_bound_max is not None, \
                "show_trg_upper_bound_max must be set for random_per_sample mode"
            assert 0.0 <= p_zero <= 1.0, f"show_trg_p_zero must be in [0, 1], got {p_zero}"
            assert draw_lower <= upper_bound_max, \
                "show_trg_draw_lower must not exceed show_trg_upper_bound_max"
        if mode == "scheduled":
            assert start is not None, \
                "show_trg_max_idx_start must be set for scheduled target mode"
            assert schedule in ("linear", "cosine"), \
                f"show_trg_schedule must be 'linear' or 'cosine', got {schedule}"
            assert 0.0 < schedule_end_frac <= 1.0, \
                f"show_trg_schedule_end_frac must be in (0, 1], got {schedule_end_frac}"
            assert 0.0 <= ramp_up_frac < 1.0, \
                f"show_trg_ramp_up_frac must be in [0, 1), got {ramp_up_frac}"
            if ramp_up_frac > 0.0:
                assert ramp_start is not None, \
                    "show_trg_ramp_start must be set when show_trg_ramp_up_frac > 0"
        if warmup_epoch is not None:
            assert isinstance(warmup_epoch, int) and warmup_epoch >= 0, \
                f"warmup_epoch must be a non-negative int or None, got {warmup_epoch}"
            if warmup_epoch > 0 and mode == "scheduled":
                warm_end = epoch_show_trg + warmup_epoch
                sched_end = int(round(schedule_end_frac * max_epochs))
                assert warm_end < sched_end, \
                    f"warmup ends at epoch {warm_end} but the schedule already " \
                    f"ends at {sched_end}"
        if warmup_bound is not None:
            assert warmup_epoch and warmup_epoch > 0, \
                "warmup_bound requires warmup_epoch > 0"
            if units == "fraction":
                assert 0.0 <= warmup_bound <= 1.0, \
                    f"warmup_bound must be in [0, 1] when show_trg_units='fraction', got {warmup_bound}"

        self.mode = mode
        self.units = units
        self.epoch_show_trg = epoch_show_trg
        self.max_epochs = max_epochs
        self.show_trg_max_idx = show_trg_max_idx
        self.upper_bound_max = upper_bound_max
        self.start = start
        self.end = end
        self.schedule = schedule
        self.schedule_end_frac = schedule_end_frac
        self.random_draw = random_draw
        self.draw_lower = draw_lower
        self.draw_lower_end = draw_lower_end
        self.draw_upper = draw_upper
        self.warmup_epoch = warmup_epoch or 0
        self.warmup_bound = warmup_bound
        self.ramp_up_frac = ramp_up_frac
        self.ramp_start = ramp_start
        self.p_zero = p_zero

    @classmethod
    def from_config(cls, config):
        """Build from the training config (all show_trg / curriculum keys)."""
        t = config["training"]
        return cls(
            mode=t.get("target_show_mode", None),
            units=t.get("show_trg_units", "index"),
            epoch_show_trg=t.get("epoch_show_trg", 0),
            max_epochs=t["max_epochs"],
            show_trg_max_idx=t.get("show_trg_max_idx", None),
            upper_bound_max=t.get("show_trg_upper_bound_max", None),
            start=t.get("show_trg_max_idx_start", None),
            end=t.get("show_trg_max_idx_end", 0),
            schedule=t.get("show_trg_schedule", "linear"),
            schedule_end_frac=t.get("show_trg_schedule_end_frac", 1.0),
            random_draw=t.get("show_trg_random_draw", True),
            draw_lower=t.get("show_trg_draw_lower", 0.0),
            draw_lower_end=t.get("show_trg_draw_lower_end", 0.0),
            draw_upper=t.get("show_trg_draw_upper", None),
            warmup_epoch=t.get("warmup_epoch", None),
            warmup_bound=t.get("warmup_bound", None),
            ramp_up_frac=t.get("show_trg_ramp_up_frac", 0.0),
            ramp_start=t.get("show_trg_ramp_start", None),
            p_zero=t.get("show_trg_p_zero", 0.0),
        )

    def _full_bound(self):
        """Bound that gives the FULL sequence (warmup): every position shown."""
        return 1.0 if self.units == "fraction" else 10 ** 9

    def _cast(self, bound):
        """Fraction mode keeps float bounds in [0, 1]; index mode rounds to int."""
        if self.units == "fraction":
            return float(bound)
        return int(round(bound))

    def scheduled_bound(self, epoch):
        """Forcing upper bound at a given epoch under the annealing schedule."""
        start_ep = self.epoch_show_trg + self.warmup_epoch
        end_ep = min(
            self.max_epochs - 1,
            int(round(self.schedule_end_frac * self.max_epochs))
        )

        # warmup: constant forcing level (warmup_bound; default full forcing),
        # skipped when warmup_epoch == 0
        if self.warmup_epoch > 0 and self.epoch_show_trg <= epoch < start_ep:
            level = self.warmup_bound if self.warmup_bound is not None \
                else self._full_bound()
            return self._cast(level)

        if epoch <= start_ep:
            # the bound is only consumed once the curriculum activates
            # (epoch == start_ep): with a ramp-up, start from
            # ramp_start instead of jumping to the full bound
            if self.ramp_up_frac > 0.0:
                return self._cast(self.ramp_start)
            return self._cast(self.start)
        if epoch >= end_ep:
            return self._cast(self.end)

        frac = (epoch - start_ep) / max(1, end_ep - start_ep)

        # Phase 1: ramp up from ramp_start to start
        ramp_frac = self.ramp_up_frac
        if ramp_frac > 0.0 and frac < ramp_frac:
            return self._cast(
                self.ramp_start + (self.start - self.ramp_start)
                * (frac / ramp_frac)
            )

        # Phase 2: anneal from start to end
        if ramp_frac > 0.0:
            frac = (frac - ramp_frac) / (1.0 - ramp_frac)
        if self.schedule == "cosine":
            w = 0.5 * (1.0 + math.cos(math.pi * frac))  # 1 -> 0
        else:  # linear
            w = 1.0 - frac

        return self._cast(self.end + (self.start - self.end) * w)

    def _draw_lower_at(self, epoch):
        """Draw-window lower end at an epoch: anneals draw_lower ->
        draw_lower_end on the same schedule (shape and timing) as the forcing
        bound. The ramp-up phase is ignored (the lower end stays at
        draw_lower until the annealing starts)."""
        if self.draw_lower == self.draw_lower_end:
            return self.draw_lower
        start_ep = self.epoch_show_trg + self.warmup_epoch
        end_ep = min(
            self.max_epochs - 1,
            int(round(self.schedule_end_frac * self.max_epochs))
        )
        if epoch <= start_ep:
            return self.draw_lower
        if epoch >= end_ep:
            return self.draw_lower_end
        frac = (epoch - start_ep) / max(1, end_ep - start_ep)
        if self.schedule == "cosine":
            w = 0.5 * (1.0 + math.cos(math.pi * frac))  # 1 -> 0
        else:  # linear
            w = 1.0 - frac
        return self.draw_lower_end + (self.draw_lower - self.draw_lower_end) * w

    def draw(self, epoch):
        """Scheduled-mode bound at an epoch, with optional uniform random draw.

        The draw window is [draw_lower(epoch), min(draw_upper, bound)]
        (draw_upper defaults to the schedule bound; draw_lower anneals to
        draw_lower_end, see _draw_lower_at). The window is clamped to the
        current schedule bound: once the annealing bound falls below the lower
        end, the window collapses and the bound itself is returned.
        """
        bound = self.scheduled_bound(epoch)
        if self.random_draw and bound > 0:
            upper = bound if self.draw_upper is None else min(self.draw_upper, bound)
            lower = min(self._draw_lower_at(epoch), upper)
            if lower >= upper:
                return upper
            if self.units == "fraction":
                bound = random.uniform(lower, upper)
            else:
                bound = random.randint(int(lower), int(upper))
        return bound

    def draw_random_mode(self):
        """Random mode: uniform draw in [0, upper_bound_max] (epoch-free)."""
        if self.units == "fraction":
            return random.uniform(0.0, self.upper_bound_max)
        return random.randint(0, self.upper_bound_max)

    def draw_per_sample(self, batch_size, device=None, generator=None):
        """random_per_sample mode: one conditioning fraction per sample.

        With probability p_zero a sample gets exactly 0 (pure forecast),
        otherwise a uniform draw in [draw_lower, upper_bound_max]. Free-run
        and conditioned samples thus share ONE batch and ONE rollout.

        Returns:
            (batch_size,) float tensor of fractions in [0, 1]
        """
        import torch
        u = torch.rand(batch_size, device=device, generator=generator)
        f = self.draw_lower + (self.upper_bound_max - self.draw_lower) * u
        zero = torch.rand(batch_size, device=device, generator=generator) < self.p_zero
        return torch.where(zero, torch.zeros_like(f), f)

