import ipdb
import numpy as np

from neurons.validator_utils import adjust_for_vtrust


def test_adjust_for_vtrust():
    rng = np.random.default_rng(seed=1337)

    for ii in range(16):
        weights = rng.uniform(0.0, 1.0, size=8)
        weights = weights / np.sum(weights)

        consensus = rng.uniform(0.0, 1.0, size=8)
        consensus = consensus / np.sum(consensus)

        vtrust_max = rng.uniform(0.5, 0.9)
        consensus = vtrust_max * consensus

        vtrust_orig = np.minimum(weights, consensus).sum()

        vtrust_min = rng.uniform(vtrust_orig + 1e-3, vtrust_max - 1e-3)

        weights_adjusted = adjust_for_vtrust(weights, consensus, vtrust_min)

        # Check that the expected vtrust is equal to vtrust_min.
        vtrust_pred = np.minimum(weights_adjusted, consensus).sum()
        np.testing.assert_allclose(vtrust_pred, vtrust_min)

        # The returned weights should be positive and sum to 1.
        assert np.all(weights_adjusted >= 0.0)
        np.testing.assert_allclose(np.sum(weights_adjusted), 1.0)


if __name__ == '__main__':
    with ipdb.launch_ipdb_on_exception():
        test_adjust_for_vtrust()