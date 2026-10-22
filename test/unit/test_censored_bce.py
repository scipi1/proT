"""
Unit tests for the weighted "alive" BCE term of CensoredLogNormalNLL
(bce_on_observed + bce_weight) and for the fail-stop scoring artifact.

Covers: backward compatibility (bce_weight=1.0 == legacy formula), weight 0
== flag off, the term only touching observed positions, anchor exclusion in
fail mode, gradient direction on a false crossing, numerical stability,
per_sample/forward consistency, and the regression documenting why stopped
("zero, sigma~0") outputs must not be scored with the NLL.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from proT.training.losses import CensoredLogNormalNLL

EPS = 1e-3
SHIFT = 1e-3
LOG_T = math.log(1.0 + SHIFT)
EXP_LEN = 6  # short expected length for the fail-mode toy sequences
NAN = float("nan")


def _toy_batch():
    """One sample, two variables, 6 positions each (expected_len = 6).

    var 1: honest failure (last observed 0.9 at pos 3, censored tail 4-6)
    var 2: survived (observed to full length, healthy, ends at 0.6)
    """
    pos = torch.tensor([[1., 2., 3., 4., 5., 6., 1., 2., 3., 4., 5., 6.]])
    var = torch.tensor([[1., 1., 1., 1., 1., 1., 2., 2., 2., 2., 2., 2.]])
    trg = torch.tensor([[0.1, 0.5, 0.9, NAN, NAN, NAN,
                         0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])
    return pos, var, trg


def _masks(trg, var, pos):
    return CensoredLogNormalNLL.classify_termination(
        trg, var, pos, expected_len=EXP_LEN, threshold=1.0, fail_margin=0.8)


def _loss(**kw):
    base = dict(shift=SHIFT, eps=EPS, expected_len=EXP_LEN, fail_anchor=1.0)
    base.update(kw)
    return CensoredLogNormalNLL(**base)


def _pred(trg, mu_val=None):
    """mu = log(target + shift) where observed (NaN -> 1.2), sigma = 0.5."""
    y = torch.nan_to_num(trg, nan=1.2)
    mu = torch.log(y + SHIFT) if mu_val is None else torch.full_like(y, mu_val)
    raw = torch.full_like(mu, math.log(math.expm1(0.5 - EPS)))
    return mu, raw


class TestBackwardCompatibility:
    def test_default_weight_matches_legacy_formula(self):
        """bce_weight=1.0 (no masks) == NLL - log_ndtr(-z) on observed."""
        _, _, trg = _toy_batch()
        mu, raw = _pred(trg)
        f = CensoredLogNormalNLL(shift=SHIFT, eps=EPS, bce_on_observed=True)
        sigma = F.softplus(raw) + EPS
        z = (mu - LOG_T) / sigma
        log_y = torch.log(torch.nan_to_num(trg) + SHIFT)
        obs = (0.5 * math.log(2 * math.pi) + torch.log(sigma)
               + 0.5 * ((log_y - mu) / sigma) ** 2 + log_y
               - torch.special.log_ndtr(-z))
        ref = torch.where(~torch.isnan(trg), obs,
                          -torch.special.log_ndtr(z)).mean()
        assert torch.allclose(f(mu, raw, trg), ref, atol=1e-6)

    def test_weight_ignored_when_flag_off(self):
        pos, var, trg = _toy_batch()
        mu, raw = _pred(trg)
        m = _masks(trg, var, pos)
        a = _loss(bce_on_observed=False, bce_weight=7.0)(mu, raw, trg, masks=m)
        b = _loss(bce_on_observed=False)(mu, raw, trg, masks=m)
        assert torch.allclose(a, b)

    def test_weight_zero_equals_flag_off(self):
        pos, var, trg = _toy_batch()
        mu, raw = _pred(trg)
        m = _masks(trg, var, pos)
        a = _loss(bce_on_observed=True, bce_weight=0.0)(mu, raw, trg, masks=m)
        b = _loss(bce_on_observed=False)(mu, raw, trg, masks=m)
        assert torch.allclose(a, b)


class TestBceScope:
    def test_only_observed_non_anchor_positions_change(self):
        """Per-position NLL differs only on observed, non-anchor positions."""
        pos, var, trg = _toy_batch()
        mu, raw = _pred(trg, mu_val=LOG_T)  # p_fail = 0.5 everywhere
        m = _masks(trg, var, pos)
        f_on = _loss(bce_on_observed=True, bce_weight=0.5)
        f_off = _loss(bce_on_observed=False)
        expected_on = m["observed"][0] & ~m["anchor"][0]
        for j in range(trg.shape[1]):
            if not m["active"][0, j]:
                continue
            mj = dict(m)
            only = torch.zeros_like(m["active"])
            only[0, j] = True
            mj["active"] = m["active"] & only
            d = float(f_on.per_sample(mu, raw, trg, mj)
                      - f_off.per_sample(mu, raw, trg, mj))
            exp = 0.5 * math.log(2.0) if expected_on[j] else 0.0
            assert d == pytest.approx(exp, abs=1e-5), f"position {j}"

    def test_anchor_excluded(self):
        pos, var, trg = _toy_batch()
        m = _masks(trg, var, pos)
        assert int(m["anchor"].sum()) == 1 and bool(m["anchor"][0, 2])
        add = _loss(bce_on_observed=True, bce_weight=1.0)._bce_add(
            torch.zeros_like(trg), m)  # z = 0 -> p_fail = 0.5
        assert float(add[0, 2]) == 0.0
        assert float(add[0, 0]) == pytest.approx(math.log(2.0), abs=1e-6)

    def test_interrupted_tail_stays_masked(self):
        pos = torch.tensor([[1., 2., 3., 4., 5., 6.]])
        var = torch.ones_like(pos)
        trg = torch.tensor([[0.1, 0.3, NAN, NAN, NAN, NAN]])  # 0.3 < 0.8
        m = _masks(trg, var, pos)
        assert not m["active"][0, 2:].any()
        mu, raw = _pred(trg)
        a = _loss(bce_on_observed=True, bce_weight=0.0).per_sample(mu, raw, trg, m)
        b = _loss(bce_on_observed=False).per_sample(mu, raw, trg, m)
        assert torch.allclose(a, b)


class TestGradientAndStability:
    def test_false_crossing_pushes_mu_down(self):
        """Healthy var 2 predicted at 1.3 (true 0.6): BCE adds downward push."""
        pos, var, trg = _toy_batch()
        m = _masks(trg, var, pos)
        grads = {}
        for flag in (False, True):
            mu, raw = _pred(trg)
            mu = mu.clone()
            mu[0, 11] = math.log(1.3 + SHIFT)
            mu.requires_grad_(True)
            _loss(bce_on_observed=flag, bce_weight=0.5)(
                mu, raw, trg, masks=m).backward()
            grads[flag] = float(mu.grad[0, 11])
        assert grads[True] > grads[False] > 0

    @pytest.mark.parametrize("mu_val", [-50.0, 0.0, 50.0])
    @pytest.mark.parametrize("raw_val", [-20.0, 0.0, 20.0])
    def test_finite_at_extremes(self, mu_val, raw_val):
        pos, var, trg = _toy_batch()
        m = _masks(trg, var, pos)
        mu = torch.full_like(trg, mu_val, requires_grad=True)
        raw = torch.full_like(trg, raw_val, requires_grad=True)
        loss = _loss(bce_on_observed=True, bce_weight=0.5)(mu, raw, trg, masks=m)
        loss.backward()
        assert torch.isfinite(loss)
        assert torch.isfinite(mu.grad).all() and torch.isfinite(raw.grad).all()

    def test_per_sample_matches_forward(self):
        pos, var, trg = _toy_batch()
        m = _masks(trg, var, pos)
        mu, raw = _pred(trg, mu_val=math.log(0.7))
        f = _loss(bce_on_observed=True, bce_weight=0.5, fail_anchor=0.0)
        assert torch.allclose(f.per_sample(mu, raw, trg, m).mean(),
                              f(mu, raw, trg, masks=m), atol=1e-6)

    def test_bce_term_diagnostic(self):
        pos, var, trg = _toy_batch()
        m = _masks(trg, var, pos)
        mu = torch.full_like(trg, LOG_T)  # p_fail = 0.5
        raw = torch.zeros_like(trg)
        f = _loss(bce_on_observed=True, bce_weight=0.5)
        assert float(f.bce_term(mu, raw, trg, m)) == pytest.approx(
            math.log(2.0), abs=1e-6)


class TestFailStopScoringArtifact:
    def test_stopped_output_explodes_nll_raw_does_not(self):
        """Why the notebook must score RAW outputs: a stopped position
        (median 0, raw_sigma -20) on a censored target costs > 1e6 nats;
        the raw rollout above the threshold costs < 1 nat."""
        f = CensoredLogNormalNLL(shift=SHIFT, eps=EPS)
        trg = torch.tensor([[NAN]])
        stopped = f.per_sample(torch.tensor([[math.log(SHIFT)]]),
                               torch.tensor([[-20.0]]), trg)
        raw = f.per_sample(torch.tensor([[math.log(1.3)]]),
                           torch.tensor([[0.0]]), trg)
        assert float(stopped) > 1e6
        assert float(raw) < 1.0
