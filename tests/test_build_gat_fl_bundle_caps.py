"""Bundle builder test-split cap semantics (no PTB-XL I/O)."""

from __future__ import annotations


def test_cap_slice_logic():
    test_recs = list(range(1838))
    max_test_cap = 200
    capped = test_recs[:max_test_cap] if max_test_cap else test_recs
    assert len(capped) == 200
    full = test_recs[: None] if False else test_recs
    assert len(full) == 1838


def test_full_test_mode_disables_cap():
    max_test_cap = None  # --full-test or default
    test_recs = [0] * 1838
    out = test_recs[:max_test_cap] if max_test_cap else test_recs
    assert len(out) == 1838
