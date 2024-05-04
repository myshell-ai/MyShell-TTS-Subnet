import bittensor as bt
import numpy as np
from scipy import optimize, stats
from scipy.special import expit


def get_p_win_sorted(m1, v1, n1, m2, v2, n2):
    # Assume 1 is old, 2 is new.
    vn1 = v1 / n1
    vn2 = v2 / n2
    with np.errstate(divide="ignore", invalid="ignore"):
        df = (vn1 + vn2) ** 2 / (vn1**2 / (n1 - 1) + vn2**2 / (n2 - 1))

    # If df is undefined, variances are zero (assumes n1 > 0 & n2 > 0).
    # Hence it doesn't matter what df is as long as it's not NaN.
    df = np.where(np.isnan(df), 1, df)
    denom = np.sqrt(vn1 + vn2)

    d = m1 - m2
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.divide(d, denom)[()]

    t_dist = stats.distributions.t(df)
    # This is the pvalue for m1 > m2. Smaller values better.
    pvalue = t_dist.sf(t)
    # Subtract to get the probability that m1 < m2.
    p_win = 1 - pvalue

    # Apply advantage to the older model.
    p_win_adv = expit(50 * (p_win - 0.90))

    # If both variances are zero due to getting the scores clipped then split it 50/50.
    if np.isnan(p_win) or denom == 0.0:
        p_win_adv = 0.5

    return p_win_adv


def get_p_win(m1, v1, n1, m2, v2, n2, block1, block2):
    if block1 < block2:
        # 1 is newer.
        p_win_2 = get_p_win_sorted(m2, v2, n2, m1, v1, n1)
        p_win_1 = 1 - p_win_2
    else:
        p_win_1 = get_p_win_sorted(m1, v1, n1, m2, v2, n2)

    return p_win_1


def compute_wins(sample_mean: np.ndarray, sample_var: np.ndarray, block: np.ndarray, num: int):
    """
    Computes the win rate for each model based on loss comparison.
    """
    p_win = np.zeros_like(sample_mean)

    n_uids = len(sample_mean)
    for uid_ii in range(n_uids):
        mean_ii, var_ii, block_ii = sample_mean[uid_ii], sample_var[uid_ii], block[uid_ii]

        for uid_jj in range(uid_ii + 1, n_uids):
            mean_jj, var_jj, block_jj = sample_mean[uid_jj], sample_var[uid_jj], block[uid_jj]

            # Compute the win rate for each pair of uids.
            p_win_ii = get_p_win(mean_ii, var_ii, num, mean_jj, var_jj, num, block_ii, block_jj)
            p_win[uid_ii] += p_win_ii
            p_win[uid_jj] += 1 - p_win_ii

    return p_win


def adjust_for_vtrust(weights: np.ndarray, consensus: np.ndarray, vtrust_min: float = 0.7):
    """
    Interpolate between the current weight and the normalized consensus weights so that the
    vtrust does not fall below vturst_min, assuming the consensus does not change.
    """
    vtrust_loss_desired = 1 - vtrust_min

    # If the predicted vtrust is already above vtrust_min, then just return the current weights.
    orig_vtrust_loss = np.maximum(0.0, weights - consensus).sum()
    if orig_vtrust_loss <= vtrust_loss_desired:
        bt.logging.info("Weights already satisfy vtrust_min. {} >= {}.", 1 - orig_vtrust_loss, vtrust_min)
        return weights

    # If maximum vtrust allowable by the current consensus is less that vtrust_min, then choose the smallest lambda
    # that still maximizes the predicted vtrust. Otherwise, find lambda that achieves vtrust_min.
    vtrust_loss_min = 1 - np.sum(consensus)
    if vtrust_loss_min > vtrust_loss_desired:
        bt.logging.info(
            "Maximum possible vtrust with current consensus is less than vtrust_min. {} < {}.".format(
                1 - vtrust_loss_min, vtrust_min
            )
        )
        vtrust_loss_desired = 1.05 * vtrust_loss_min

    # We could solve this with a LP, but just do rootfinding with scipy.
    consensus_normalized = consensus / np.sum(consensus)

    def fn(lam: float):
        new_weights = (1 - lam) * weights + lam * consensus_normalized
        vtrust_loss = np.maximum(0.0, new_weights - consensus).sum()
        return vtrust_loss - vtrust_loss_desired

    sol = optimize.root_scalar(fn, bracket=[0, 1], method="brentq")
    lam_opt = sol.root

    new_weights = (1 - lam_opt) * weights + lam_opt * consensus_normalized
    vtrust_pred = np.minimum(weights, consensus).sum()
    bt.logging.info("Interpolated weights to satisfy vtrust_min. {} -> {}.", 1 - orig_vtrust_loss, vtrust_pred)
    return new_weights


class EvalQueue:
    """Class for managing the order and frequency of evaluating models.

    Evaluation is split into epochs, where each epoch validates some subset of all the models using the same seed.
    Importantly, the weights are only updated at the end of each epoch.

    Each epoch, a total of 32 models are evaluated. Of these 32, we pick
    - The top 8 models based on the current weights.
    - 16 models randomly sampled with probability proportional to their current rank.
    - The 8 models that have not been evaluated for the longest time.
      This guarantees that each model will be evaluated at least once every 32 epochs.

    Except for the first epoch, where we evaluate all models to get some estimaate of how they perform.
    """

    def __init__(self, weights: np.ndarray):
        self.n_models = len(weights)
        self._weights = weights
        self.rng = np.random.default_rng()
        self.age_queue = self.rng.choice(self.n_models, self.n_models, replace=False).tolist()
        self.seed, self.queue = self._get_shuffled_init()
        self.epochs = 0

    @property
    def epoch_is_done(self):
        return len(self.queue) == 0

    def update_weights(self, weights: np.ndarray):
        self._weights = weights

    def _select_model(self, uid: int):
        """Place it at the end of the age_queue."""
        self.age_queue.remove(uid)
        self.age_queue.append(uid)

    def _get_shuffled_init(self) -> tuple[int, list]:
        seed = self.rng.integers(0, 2**16)
        return seed, self.rng.choice(self.n_models, self.n_models, replace=False).tolist()

    def _get_shuffled(self) -> tuple[int, list]:
        # Sample random seed.
        seed = self.rng.integers(0, 2**16)

        # Top 8 models based on the current weights.
        idxs = np.argsort(self._weights)[::-1]
        top_8 = idxs[:8]
        is_top_8 = np.zeros(self.n_models, dtype=bool)
        is_top_8[top_8] = True
        for uid in top_8:
            self._select_model(uid)

        # 16 models randomly sampled with probability using their current rank.
        ranks = np.zeros(self.n_models)
        ranks[idxs] = np.arange(self.n_models)
        probs = np.exp(-ranks / 32)
        #    Don't sample the top 8.
        probs[is_top_8] = 0
        probs /= probs.sum()
        random_16 = self.rng.choice(self.n_models, 16, p=probs, replace=False)
        for uid in random_16:
            self._select_model(uid)

        # The 8 models that have not been evaluated for the longest time.
        age_8 = self.age_queue[:8]
        for uid in age_8:
            self._select_model(uid)

        uids = top_8.tolist() + random_16.tolist() + age_8
        return seed, uids

    def take(self, n: int):
        seeds = []
        uids = []
        # Don't start a new epoch in the middle.
        if len(self.queue) > 0:
            n = min(n, len(self.queue))

        for _ in range(n):
            seed, uid = self.next()
            seeds.append(seed)
            uids.append(uid)
        return seeds, uids

    def take_all(self):
        return self.take(len(self.queue))

    def next(self):
        if len(self.queue) == 0:
            self.seed, self.queue = self._get_shuffled()
            self.epochs += 1
        return self.seed, self.queue.pop()
