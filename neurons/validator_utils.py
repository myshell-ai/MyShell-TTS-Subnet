from typing import Optional, Tuple, Union

import bittensor as bt
import numpy as np
import torch
from bittensor.extrinsics.set_weights import set_weights_extrinsic
from scipy import optimize, stats
import typing
import constants

def iswin(score_i, score_j, block_i, block_j):
    """
    Determines the winner between two models based on the epsilon adjusted score.

    Parameters:
        score_i (float): Score of uid i on batch
        score_j (float): Score of uid j on batch.
        block_i (int): Block of uid i.
        block_j (int): Block of uid j.
    Returns:
        bool: True if score i is better, False otherwise.
    """
    # Adjust score based on timestamp and pretrain epsilon
    score_i = (1 - constants.timestamp_epsilon) * score_i if block_i > block_j else score_i
    score_j = (1 - constants.timestamp_epsilon) * score_j if block_j > block_i else score_j
    return score_i > score_j

def compute_wins(
    uids: typing.List[int],
    scores_per_uid: typing.Dict[int, typing.List[float]],
    block: np.ndarray,
):
    """
    Computes the wins and win rate for each model based on score comparison.

    Parameters:
        uids (list): A list of uids to compare.
        scores_per_uid (dict): A dictionary of scores for each uid by batch.
        batches (List): A list of data batches.
        uid_to_block (dict): A dictionary of blocks for each uid.

    Returns:
        tuple: A tuple containing two dictionaries, one for wins and one for win rates.
    """
    blacklist_uids = []
    for i, uid_i in enumerate(uids):
        block_i = block[uid_i]
        # if scores are all 0
        if uid_i in blacklist_uids:
            continue
        if all([score == 0 for score in scores_per_uid[uid_i]]):
            blacklist_uids.append(uid_i)
            continue
        for j, uid_j in enumerate(uids):
            if i == j or uid_j in blacklist_uids:
                continue
            block_j = block[uid_j]
            # check if scores are similar...
            # if the scores are similar, we will blacklist the model with the higher block number
            if abs(np.asarray(scores_per_uid[uid_i]) - np.asarray(scores_per_uid[uid_j])).max() < 5e-2:
                if block_i < block_j:
                    blacklist_uids.append(uid_j)
                else:
                    if block_i > block_j: # miners should use --use_hotkey_in_hash to avoid the same block clones
                        blacklist_uids.append(uid_i)
                        break
    whitelist_uids = [uid for uid in uids if uid not in blacklist_uids]

    wins = {uid: 0 for uid in uids}
    win_rate_1 = {uid: 0 for uid in uids}
    for i, uid_i in enumerate(whitelist_uids):
        total_matches = 0
        block_i = block[uid_i]
        for j, uid_j in enumerate(whitelist_uids):
            if i == j:
                continue
            block_j = block[uid_j]
            batches_i = len(scores_per_uid[uid_i])
            batches_j = len(scores_per_uid[uid_j])
            for batch_idx in range(0, min(batches_i, batches_j)):
                scores_i = scores_per_uid[uid_i][batch_idx]
                scores_j = scores_per_uid[uid_j][batch_idx]
                wins[uid_i] += 1 if iswin(scores_i, scores_j, block_i, block_j) else 0
                total_matches += 1
        # Calculate win rate for uid i
        win_rate_1[uid_i] = wins[uid_i] / total_matches if total_matches > 0 else 0
    wins = {uid: 0 for uid in uids}
    win_rate_4 = {uid: 0 for uid in uids}
    for i, uid_i in enumerate(whitelist_uids):
        total_matches = 0
        block_i = block[uid_i]
        for j, uid_j in enumerate(whitelist_uids):
            if i == j:
                continue
            block_j = block[uid_j]
            batches_i = len(scores_per_uid[uid_i])//4
            batches_j = len(scores_per_uid[uid_j])//4
            for batch_idx in range(0, min(batches_i, batches_j)):
                scores_is = scores_per_uid[uid_i][batch_idx * 4: (batch_idx + 1) * 4]
                scores_js = scores_per_uid[uid_j][batch_idx * 4: (batch_idx + 1) * 4]
                wins[uid_i] += 1 if iswin(np.mean(scores_is), np.mean(scores_js), block_i, block_j) else 0
                total_matches += 1
        # Calculate win rate for uid i
        win_rate_4[uid_i] = wins[uid_i] / total_matches if total_matches > 0 else 0
    wins = {uid: 0 for uid in uids}
    win_rate_8 = {uid: 0 for uid in uids}
    for i, uid_i in enumerate(whitelist_uids):
        total_matches = 0
        block_i = block[uid_i]
        for j, uid_j in enumerate(whitelist_uids):
            if i == j:
                continue
            block_j = block[uid_j]
            batches_i = len(scores_per_uid[uid_i])//8
            batches_j = len(scores_per_uid[uid_j])//8
            temp_scores_i = scores_per_uid[uid_i][1::2] + scores_per_uid[uid_i][::2]
            temp_scores_j = scores_per_uid[uid_j][1::2] + scores_per_uid[uid_j][::2]
            temp_scores_i.reverse()
            temp_scores_j.reverse()
            for batch_idx in range(0, min(batches_i, batches_j)):
                scores_is = temp_scores_i[batch_idx * 8: (batch_idx + 1) * 8]
                scores_js = temp_scores_j[batch_idx * 8: (batch_idx + 1) * 8]
                wins[uid_i] += 1 if iswin(np.mean(scores_is), np.mean(scores_js), block_i, block_j) else 0
                total_matches += 1
        # Calculate win rate for uid i
        win_rate_8[uid_i] = wins[uid_i] / total_matches if total_matches > 0 else 0

    weights = {win_rate_1: 0.25, win_rate_4: 0.5, win_rate_8: 1.0}
    weights_sum = sum(weights.values())
    weights = {winrate_dict: weight / weights_sum for winrate_dict, weight in weights.items()}
    win_rate = {uid: sum([win_rate_dict[uid] * weight for win_rate_dict, weight in weights.items()]) for uid in uids}
    return wins, win_rate

def adjust_for_vtrust(weights: np.ndarray, consensus: np.ndarray, vtrust_min: float = 0.5):
    """
    Interpolate between the current weight and the normalized consensus weights so that the
    vtrust does not fall below vturst_min, assuming the consensus does not change.
    """
    vtrust_loss_desired = 1 - vtrust_min

    # If the predicted vtrust is already above vtrust_min, then just return the current weights.
    orig_vtrust_loss = np.maximum(0.0, weights - consensus).sum()
    if orig_vtrust_loss <= vtrust_loss_desired:
        bt.logging.info("Weights already satisfy vtrust_min. {} >= {}.".format(1 - orig_vtrust_loss, vtrust_min))
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
    bt.logging.info("Interpolated weights to satisfy vtrust_min. {} -> {}.".format(1 - orig_vtrust_loss, vtrust_pred))
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


def set_weights_with_err_msg(
    self: bt.subtensor,
    wallet: bt.wallet,
    netuid: int,
    uids: [torch.LongTensor, list],
    weights: Union[torch.FloatTensor, list],
    version_key: int = bt.__version_as_int__,
    uid: Optional[int] = None,
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = False,
    prompt: bool = False,
    max_retries: int = 5,
) -> Tuple[bool, str, list[Exception]]:
    """Same as subtensor.set_weights, but with additional error messages."""
    uid = self.get_uid_for_hotkey_on_subnet(wallet.hotkey.ss58_address, netuid)
    retries = 0
    success = False
    message = "No attempt made. Perhaps it is too soon to set weights!"
    exceptions = []

    while (
        self.blocks_since_last_update(netuid, uid) > self.weights_rate_limit(netuid)  # type: ignore
        and retries < max_retries
    ):
        try:
            success, message = set_weights_extrinsic(
                subtensor=self,
                wallet=wallet,
                netuid=netuid,
                uids=uids,
                weights=weights,
                version_key=version_key,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
                prompt=prompt,
            )

            if (wait_for_inclusion or wait_for_finalization) and success:
                return success, message, exceptions

        except Exception as e:
            bt.logging.exception(f"Error setting weights: {e}")
            exceptions.append(e)
        finally:
            retries += 1

    return success, message, exceptions
