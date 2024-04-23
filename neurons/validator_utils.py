import numpy as np
from scipy import stats
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
