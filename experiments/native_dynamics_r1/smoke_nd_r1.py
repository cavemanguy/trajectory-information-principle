import math
import tempfile

import numpy as np
import torch

import run_nd_r1 as nd


def main():
    # Code-path validation only. This script does not train/evaluate a primary seed.
    nd.PAIR_N = 16
    nd.BOOT_N = 20

    nd.set_seed(12345)
    model = nd.Core().eval()

    y = nd.make_memories(64, 101)
    p = nd.make_perms(64, 102)
    tr = nd.generate_trajectory(model, y, p)
    assert tr.shape == (64, nd.STEPS + 1, nd.LATENT)

    g = nd.geometry(tr)
    assert g["speed"].shape == (64, nd.STEPS)
    assert g["direction"].shape == (64, nd.STEPS, nd.LATENT)
    assert g["turn_cos"].shape == (64, nd.STEPS - 1)

    bank, _ = nd.make_pair_bank(13)
    assert len(bank) == nd.N_REL
    for r, (base, alt, perms) in enumerate(bank):
        diff = base != alt
        assert torch.all(diff.sum(1) == 1)
        assert torch.all(diff[:, r])
        assert perms.shape == (nd.PAIR_N, nd.N_REL)

    surv = nd.pair_survival_for_model(model, bank)
    assert surv.shape == (nd.N_REL, nd.PAIR_N, nd.STEPS + 1)
    assert np.all(np.isfinite(surv))
    assert np.max(np.abs(surv[:, :, 0] - 1.0)) < 1e-4

    train_y = nd.make_memories(128, 201)
    test_y = nd.make_memories(64, 202)
    train_p = nd.make_perms(128, 203)
    test_p = nd.make_perms(64, 204)
    train_tr = nd.generate_trajectory(model, train_y, train_p)
    test_tr = nd.generate_trajectory(model, test_y, test_p)
    access = nd.ridge_accessibility(train_tr, test_tr, train_y, test_y)
    assert len(access["state_by_time"]) == nd.STEPS + 1
    assert set(access["direction_summaries"]) == {
        "first_direction",
        "final_direction",
        "integrated_direction",
    }

    s0 = surv.copy()
    s1 = surv.copy()
    # Deterministically introduce relation heterogeneity only to exercise bootstrap code.
    for r in range(nd.N_REL):
        s1[r, :, -1] *= 1.0 + 0.03 * r
    boot = nd.bootstrap_delta_G(13, s0, s1)
    assert boot["n_bootstrap"] == nd.BOOT_N
    assert len(boot["ci95_percentile"]) == 2
    assert all(math.isfinite(x) for x in boot["ci95_percentile"])

    print("ND-R1 smoke checks passed")


if __name__ == "__main__":
    main()
