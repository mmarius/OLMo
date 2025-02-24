import pytest

from olmo.optim import BoltOnWarmupScheduler, LinearWithWarmup, ConstantWithDecayWithWarmupScheduler


def test_linear_with_warmup_scheduler():
    initial_lr = 1.0
    max_steps = 10_000
    scheduler = LinearWithWarmup(
        grad_clip_warmup_steps=None, grad_clip_warmup_factor=None, warmup_steps=2000, warmup_min_lr=None
    )
    assert scheduler.get_lr(initial_lr, 0, max_steps) == 0.1
    assert scheduler.get_lr(initial_lr, 2000, max_steps) == 1.0
    assert scheduler.get_lr(initial_lr, 10_000, max_steps) == 0.1
    assert scheduler.get_lr(initial_lr, 3_000, max_steps) > scheduler.get_lr(initial_lr, 5_000, max_steps)


def test_bolt_on_warmup_scheduler():
    initial_lr = 1.0
    max_steps = 11_000
    alpha_f = 0.1
    scheduler = LinearWithWarmup(
        grad_clip_warmup_steps=None,
        grad_clip_warmup_factor=None,
        warmup_steps=1000,
        alpha_f=alpha_f,
        warmup_min_lr=None,
    )
    scheduler2 = BoltOnWarmupScheduler.wrap(scheduler, 5000, 6000)
    assert scheduler.get_lr(initial_lr, 100, max_steps) > 0.0
    assert scheduler2.get_lr(initial_lr, 100, max_steps) == 0.0
    assert scheduler2.get_lr(initial_lr, 5000, max_steps) == 0.0
    assert scheduler2.get_lr(initial_lr, 5500, max_steps) == pytest.approx(0.25 * (1 + alpha_f))
    assert scheduler2.get_lr(initial_lr, 6000, max_steps) == pytest.approx(0.5 * (1 + alpha_f))
    assert scheduler2.get_lr(initial_lr, 7000, max_steps) == scheduler.get_lr(initial_lr, 7000, max_steps)


def test_constant_with_decay_with_warmup_scheduler_sqrt_decay():
    initial_lr = 1.0
    max_steps = 10_000
    scheduler = ConstantWithDecayWithWarmupScheduler(
        grad_clip_warmup_steps=None,
        grad_clip_warmup_factor=None,
        warmup_min_lr=0.1,
        warmup_steps=2000,
        decay_steps=3000,
        decay_type="sqrt",
    )

    # Test warmup phase
    assert scheduler.get_lr(initial_lr, 0, max_steps) == 0.1  # Start at warmup_min_lr
    assert scheduler.get_lr(initial_lr, 2000, max_steps) == 1.0  # End of warmup

    # Test constant phase
    assert scheduler.get_lr(initial_lr, 3000, max_steps) == 1.0
    assert scheduler.get_lr(initial_lr, 6000, max_steps) == 1.0

    # Test decay phase
    assert scheduler.get_lr(initial_lr, 7000, max_steps) == 1.0  # Start of decay
    decay_mid = scheduler.get_lr(initial_lr, 8500, max_steps)
    decay_end = scheduler.get_lr(initial_lr, 10000, max_steps)
    assert decay_mid < 1.0  # Should have decayed
    assert decay_end < decay_mid  # Should be monotonically decreasing


def test_constant_with_decay_with_warmup_scheduler_linear_decay():
    initial_lr = 1.0
    max_steps = 10_000
    scheduler = ConstantWithDecayWithWarmupScheduler(
        grad_clip_warmup_steps=None,
        grad_clip_warmup_factor=None,
        warmup_min_lr=0.1,
        warmup_steps=2000,
        decay_steps=3000,
        decay_type="linear",
    )

    # Test linear decay behavior
    decay_start = max_steps - 3000  # 7000
    decay_mid = decay_start + 1500  # 8500
    assert scheduler.get_lr(initial_lr, decay_start, max_steps) == 1.0
    assert scheduler.get_lr(initial_lr, decay_mid, max_steps) == 0.5  # Should be halfway through decay
    assert scheduler.get_lr(initial_lr, max_steps, max_steps) == 0.0  # Should decay to 0


def test_constant_with_decay_with_warmup_scheduler_cosine_decay():
    initial_lr = 1.0
    max_steps = 10_000
    scheduler = ConstantWithDecayWithWarmupScheduler(
        grad_clip_warmup_steps=None,
        grad_clip_warmup_factor=None,
        warmup_min_lr=0.1,
        warmup_steps=2000,
        decay_steps=3000,
        decay_type="cosine",
    )

    # Test cosine decay behavior
    decay_start = max_steps - 3000  # 7000
    decay_mid = decay_start + 1500  # 8500
    assert scheduler.get_lr(initial_lr, decay_start, max_steps) == 1.0
    assert (
        abs(scheduler.get_lr(initial_lr, decay_mid, max_steps) - 0.5) < 1e-5
    )  # Should be close to 0.5 at midpoint
    assert scheduler.get_lr(initial_lr, max_steps, max_steps) == 0.0  # Should decay to 0


def test_constant_with_decay_with_warmup_scheduler_warmup_behavior():
    initial_lr = 1.0
    max_steps = 10_000
    warmup_steps = 2000
    scheduler = ConstantWithDecayWithWarmupScheduler(
        grad_clip_warmup_steps=None,
        grad_clip_warmup_factor=None,
        warmup_min_lr=0.1,
        warmup_steps=warmup_steps,
        decay_steps=3000,
        decay_type="sqrt",
    )

    # Test warmup progression
    lr_start = scheduler.get_lr(initial_lr, 0, max_steps)
    lr_quarter = scheduler.get_lr(initial_lr, warmup_steps // 4, max_steps)
    lr_half = scheduler.get_lr(initial_lr, warmup_steps // 2, max_steps)
    lr_end = scheduler.get_lr(initial_lr, warmup_steps, max_steps)

    assert lr_start == 0.1
    assert 0.1 < lr_quarter < lr_half < lr_end
    assert lr_end == initial_lr


def test_constant_with_decay_with_warmup_scheduler_monotonicity():
    initial_lr = 1.0
    max_steps = 10_000
    scheduler = ConstantWithDecayWithWarmupScheduler(
        grad_clip_warmup_steps=None,
        grad_clip_warmup_factor=None,
        warmup_min_lr=0.1,
        warmup_steps=2000,
        decay_steps=3000,
        decay_type="sqrt",
    )

    # Test monotonicity in warmup phase
    assert scheduler.get_lr(initial_lr, 500, max_steps) < scheduler.get_lr(initial_lr, 1000, max_steps)

    # Test constant phase
    assert scheduler.get_lr(initial_lr, 3000, max_steps) == scheduler.get_lr(initial_lr, 4000, max_steps)

    # Test monotonicity in decay phase
    assert scheduler.get_lr(initial_lr, 8000, max_steps) > scheduler.get_lr(initial_lr, 9000, max_steps)
