# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 const

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import datetime as dt

import tqdm
import logging
import json
import time
import random

import numpy as np
import shutil
import asyncio
import argparse
from threadpoolctl import threadpool_limits

import wandb
from model.data import ModelMetadata
from model.model_tracker import ModelTracker
from model.model_updater import ModelUpdater
from model.storage.chain.chain_model_metadata_store import ChainModelMetadataStore
from model.storage.disk.disk_model_store import DiskModelStore
from model.storage.disk.utils import get_hf_download_path
from model.storage.hugging_face.hugging_face_model_store import HuggingFaceModelStore
import threading
import multiprocessing
from rich.table import Table
from rich.console import Console

from neurons.validator_utils import compute_wins
from utilities.miner_iterator import MinerIterator
from utilities import utils
from utilities.perf_monitor import PerfMonitor
import tts_subnet
from tts_rater.rater import rate

import math
import torch
import typing
import constants
import traceback
import bittensor as bt

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def best_uid(metagraph: bt.metagraph) -> int:
    """Returns the best performing UID in the metagraph."""
    return max(range(metagraph.n), key=lambda uid: metagraph.I[uid].item())


def nearest_tempo(start_block, tempo, block):
    start_num = start_block + tempo
    intervals = (block - start_num) // tempo
    nearest_num = start_num + intervals * tempo
    if nearest_num >= block:
        nearest_num -= tempo
    return nearest_num


class RandomQueue:
    def __init__(self, items: np.ndarray):
        self.items = items
        self.rng = np.random.default_rng()
        self.seed, self.queue = self._get_shuffled()
        self.epochs = 0

    @property
    def epoch_is_done(self):
        return len(self.queue) == 0

    def _get_shuffled(self) -> tuple[int, list]:
        seed = self.rng.integers(0, 2 ** 16)
        return seed, self.rng.choice(self.items, len(self.items), replace=False).tolist()

    def take(self, n: int):
        seeds = []
        uids = []
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


class Validator:
    @staticmethod
    def config():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--device",
            type=str,
            default="cuda",
            help="Device name.",
        )
        parser.add_argument(
            "--wandb_project",
            help="Turn on wandb logging (and log to this project)",
        )
        parser.add_argument(
            "--wandb_entity",
            help="wandb entity for logging (if --wandb_project set)",
        )
        parser.add_argument(
            "--wandb_max_steps_per_run",
            type=int,
            help="number of steps before creating a new wandb run",
        )
        parser.add_argument(
            "--blocks_per_epoch",
            type=int,
            default=100,
            help="Number of blocks to wait before setting weights.",
        )
        parser.add_argument(
            "--sample_min",
            type=int,
            default=15,
            help="Number of uids to eval each step.",
        )
        parser.add_argument(
            "--dont_set_weights",
            action="store_true",
            help="Validator does not set weights on the chain.",
        )
        parser.add_argument(
            "--offline",
            action="store_true",
            help="Does not launch a wandb run, does not set weights, does not check that your key is registered.",
        )
        parser.add_argument(
            "--model_dir",
            default=os.path.join(constants.ROOT_DIR, "model-store/"),
            help="Where to store downloaded models",
        )
        parser.add_argument(
            "--netuid", type=str, default=constants.SUBNET_UID, help="The subnet UID."
        )
        parser.add_argument(
            "--genesis",
            action="store_true",
            help="Don't sync to consensus, rather start evaluation from scratch",
        )
        parser.add_argument(
            "--dtype",
            type=str,
            default="bfloat16",
            help="datatype to load model in, either bfloat16 or float16",
        )
        parser.add_argument(
            "--clean_period_minutes",
            type=int,
            default=1,
            help="How often to delete unused models",
        )
        parser.add_argument(
            "--update_delay_minutes",
            type=int,
            default=5,
            help="Period between checking for new models from each UID",
        )
        parser.add_argument(
            "--do_sample",
            action="store_true",
            help="Sample a response from each model (for leaderboard)",
        )
        parser.add_argument(
            "--num_samples_per_eval",
            type=int,
            default=96,
            help="Number of samples to evaluate per UID",
        )
        parser.add_argument(
            "--batch_size",
            type=int,
            default=16,
            help="Batch size to use for validation",
        )

        bt.subtensor.add_args(parser)
        bt.logging.add_args(parser)
        bt.wallet.add_args(parser)
        bt.axon.add_args(parser)
        config = bt.config(parser)
        return config

    def state_path(self) -> str:
        """
        Constructs a file path for storing validator state.

        Returns:
        str: A string representing the file path.
        """
        return os.path.expanduser(
            "{}/{}/{}/netuid{}/{}".format(
                bt.logging.config().logging.logging_dir,
                self.wallet.name,
                self.wallet.hotkey_str,
                self.config.netuid,
                "vali-state",
            )
        )

    def __init__(self):
        self.config = Validator.config()
        if self.config.logging_dir is None:
            self.config.logging_dir = "."
        bt.logging(config=self.config)
        bt.logging.on()

        for name, logger in logging.root.manager.loggerDict.items():
            if isinstance(logger, logging.Logger):
                if "huggingface_hub" in name:
                    logger.setLevel(logging.INFO)
                if "transformers" in name:
                    logger.setLevel(logging.WARNING)
                if name == "huggingface_hub.file_download":
                    # To remove the progress bars from downloading hf models.
                    print("name: {}".format(name))
                    logger.getEffectiveLevel = lambda: logging.NOTSET

        bt.logging.info(f"Starting validator with config: {self.config}")

        # === Bittensor objects ====
        self.wallet = bt.wallet(config=self.config)
        self.subtensor = bt.subtensor(config=self.config)
        self.dendrite = bt.dendrite(wallet=self.wallet)
        self.metagraph: bt.metagraph = self.subtensor.metagraph(self.config.netuid)
        torch.backends.cudnn.benchmark = True

        # Dont check registration status if offline.
        if not self.config.offline:
            self.uid = utils.assert_registered(self.wallet, self.metagraph)

        # Track how may run_steps this validator has completed.
        self.run_step_count = 0

        # Dont log to wandb if offline.
        if not self.config.offline and self.config.wandb_project:
            self.new_wandb_run()

        # === Running args ===
        self.weights = torch.zeros_like(self.metagraph.S)
        self.epoch_step = 0
        self.global_step = 0
        self.last_epoch = self.metagraph.block.item()

        self.ema_alpha = 0.9
        self.sample_mean_per_uid = np.zeros_like(self.weights, dtype=np.float64)
        self.sample_var_per_uid = np.zeros_like(self.weights, dtype=np.float64)
        self.count_per_uid = np.zeros_like(self.weights, dtype=np.int64)

        # Keep track of which block this uid last updated their model.
        # Default to an infinite block if we can't retrieve the metadata for the miner.
        self.block = np.full_like(self.weights, np.iinfo(np.int64).max, dtype=np.int64)

        self.uid_last_checked = {}
        self.uids_to_eval: typing.Dict[str, typing.List] = {}
        self.rng = np.random.default_rng()

        # Setup a model tracker to track which miner is using which model id.
        self.model_tracker = ModelTracker()

        # Setup a miner iterator to ensure we update all miners.
        # This subnet does not differentiate between miner and validators so this is passed all uids.
        self.miner_iterator = MinerIterator(self.metagraph.uids.tolist())

        # Setup a ModelMetadataStore
        self.metadata_store = ChainModelMetadataStore(
            self.subtensor, self.config.netuid, self.wallet
        )

        # Setup a RemoteModelStore
        self.remote_store = HuggingFaceModelStore()

        # Setup a LocalModelStore
        self.local_store = DiskModelStore(base_dir=self.config.model_dir)

        # Setup a model updater to download models as needed to match the latest provided miner metadata.
        self.model_updater = ModelUpdater(
            metadata_store=self.metadata_store,
            remote_store=self.remote_store,
            local_store=self.local_store,
            model_tracker=self.model_tracker,
        )

        # Sync to consensus
        bt.logging.trace("Pulling competition ids for all hotkeys")
        competition_ids: typing.Dict[int, typing.Optional[str]] = {}
        for uid, hotkey in enumerate(list(self.metagraph.hotkeys)):
            competition_ids[uid] = constants.ORIGINAL_COMPETITION_ID

        self.weights.copy_(self.metagraph.C)

        for competition in constants.COMPETITION_SCHEDULE:
            bt.logging.trace(
                f"Building consensus state for competition {competition.competition_id}"
            )
            # Evaluate all models on the first iteration.
            consensus = [i for i, val in enumerate(list(self.metagraph.consensus)) if competition_ids[i] == competition.competition_id]
            self.uids_queue = RandomQueue(np.array(consensus))
            self.seeds_to_eval, uids_to_eval = self.uids_queue.take(16)
            self.uids_to_eval[competition.competition_id] = uids_to_eval

            consensus_map = {uid: self.weights[uid].item() for uid in consensus}
            bt.logging.info(
                f"Consensus for competition {competition.competition_id}: {consensus_map}"
            )

            # Sync the first few models, we can sync the rest while running.
            bt.logging.info("Syncing models for the first 16 hotkeys. This may take a while...")
            uids_to_sync = list(uids_to_eval)[:16]
            hotkeys = [self.metagraph.hotkeys[uid] for uid in uids_to_sync]
            results = asyncio.run(self.model_updater.sync_models(hotkeys))
            bt.logging.info("Syncing models for the first 16 hotkeys. This may take a while... Done!")

            for uid in uids_to_sync:
                self.uid_last_checked[uid] = dt.datetime.now()

            for ii, uid in enumerate(uids_to_sync):
                hotkey = self.metagraph.hotkeys[uid]
                result = results[ii]
                if isinstance(result, Exception):
                    bt.logging.warning(
                        f"Unable to sync model for consensus UID {uid} with hotkey {hotkey}. Exception {result}"
                    )

                if (
                        self.model_tracker.get_model_metadata_for_miner_hotkey(
                            hotkey
                        )
                        is None
                ):
                    is_vali = self.metagraph.Tv[uid] > 0
                    if not is_vali:
                        bt.logging.warning(f"Unable to get metadata for consensus UID {uid} with hotkey {hotkey}")

        # Touch all models, starting a timer for them to be deleted if not used
        self.model_tracker.touch_all_miner_models()

        # == Initialize the update thread ==
        self.stop_event = threading.Event()
        self.update_thread = threading.Thread(
            target=self.update_models,
            args=(self.config.update_delay_minutes,),
            daemon=True,
        )
        self.update_thread.start()

        # == Initialize the cleaner thread to remove outdated models ==
        self.clean_thread = threading.Thread(
            target=self.clean_models,
            args=(self.config.clean_period_minutes,),
            daemon=True,
        )
        self.clean_thread.start()

    @property
    def sample_stats_corrected(self):
        count = self.count_per_uid
        denom = 1 - self.ema_alpha ** count
        safe_denom = np.where(denom == 0, 1, denom)
        sample_mean_corrected = np.where(count == 0, -1, self.sample_mean_per_uid / safe_denom)
        sample_var_corrected = np.where(count == 0, 1e-6, self.sample_var_per_uid / safe_denom)
        return sample_mean_corrected, sample_var_corrected

    def __del__(self):
        if hasattr(self, "stop_event"):
            self.stop_event.set()
            self.update_thread.join()
            self.clean_thread.join()

    def new_wandb_run(self):
        """Creates a new wandb run to save information to."""
        # Create a unique run id for this run.
        run_id = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        name = "validator-" + str(self.uid) + "-" + run_id
        self.wandb_run = wandb.init(
            name=name,
            project=self.config.wandb_project,
            entity=self.config.wandb_entity,
            config={
                "uid": self.uid,
                "hotkey": self.wallet.hotkey.ss58_address,
                "run_name": run_id,
                "version": tts_subnet.__version__,
                "type": "validator",
            },
            allow_val_change=True,
        )

        bt.logging.debug(f"Started a new wandb run: {name}")

    def update_models(self, update_delay_minutes):
        # Track how recently we updated each uid
        uid_last_checked = self.uid_last_checked

        # The below loop iterates across all miner uids and checks to see
        # if they should be updated.
        while not self.stop_event.is_set():
            try:
                # Get the next uid to check
                next_uid = next(self.miner_iterator)

                # Confirm that we haven't checked it in the last `update_delay_minutes` minutes.
                time_diff = (
                    dt.datetime.now() - uid_last_checked[next_uid]
                    if next_uid in uid_last_checked
                    else None
                )

                if time_diff and time_diff < dt.timedelta(minutes=update_delay_minutes):
                    # If we have seen it within `update_delay_minutes` minutes then sleep until it has been at least `update_delay_minutes` minutes.
                    time_to_sleep = (
                        dt.timedelta(minutes=update_delay_minutes) - time_diff
                    ).total_seconds()
                    bt.logging.trace(
                        f"Update loop has already processed all UIDs in the last {update_delay_minutes} minutes. Sleeping {time_to_sleep} seconds."
                    )
                    time.sleep(time_to_sleep)

                uid_last_checked[next_uid] = dt.datetime.now()
                bt.logging.trace(f"Updating model for UID={next_uid}")

                # Get their hotkey from the metagraph.
                hotkey = self.metagraph.hotkeys[next_uid]

                # Compare metadata and tracker, syncing new model from remote store to local if necessary.
                updated = asyncio.run(self.model_updater.sync_model(hotkey))

                if updated:
                    metadata = self.model_tracker.get_model_metadata_for_miner_hotkey(
                        hotkey
                    )
                    if metadata is not None:
                        bt.logging.trace(
                            f"Updated model for UID={next_uid}. Was new = {updated}"
                        )
                    else:
                        bt.logging.warning(
                            f"Unable to sync model for consensus UID {next_uid} with hotkey {hotkey}"
                        )

            except TimeoutError as e:
                bt.logging.warning(f"Timeout trying to sync metagraph. This is normal when the chain is congested: {e}")
            except Exception as e:
                bt.logging.error(f"Error in update loop: {e}")

        bt.logging.info("Exiting update models loop.")

    def clean_models(self, clean_period_minutes: int):
        # The below loop checks to clear out all models in local storage that are no longer referenced.
        while not self.stop_event.is_set():
            try:
                old_models = self.model_tracker.get_and_clear_old_models()

                if len(old_models) > 0:
                    bt.logging.info("Starting cleanup of stale models. Removing {}...".format(len(old_models)))

                uids_removed = []
                for hotkey, model_metadata in old_models:
                    local_path = self.local_store.get_path(hotkey)
                    model_dir = get_hf_download_path(local_path, model_metadata.id)
                    shutil.rmtree(model_dir, ignore_errors=True)

                    if hotkey in self.metagraph.hotkeys:
                        uid = self.metagraph.hotkeys.index(hotkey)
                        uids_removed.append(uid)
                    else:
                        repo_name = "{}/{}".format(model_metadata.id.namespace, model_metadata.id.name)
                        uids_removed.append(repo_name)

                if len(old_models) > 0:
                    bt.logging.info("Removed {} uids  -  {}".format(len(uids_removed), uids_removed))

            except Exception as e:
                bt.logging.error(f"Error in clean loop: {e}")
                print(traceback.format_exc())

            time.sleep(dt.timedelta(minutes=clean_period_minutes).total_seconds())

        bt.logging.info("Exiting clean models loop.")

    async def try_set_weights(self, ttl: int):
        async def _try_set_weights():
            try:
                self.weights.nan_to_num(0.0)
                self.subtensor.set_weights(
                    netuid=self.config.netuid,
                    wallet=self.wallet,
                    uids=self.metagraph.uids,
                    weights=self.weights,
                    wait_for_inclusion=False,
                    version_key=constants.weights_version_key,
                )
            except:
                pass
            ws, ui = self.weights.topk(len(self.weights))
            table = Table(title="All Weights")
            table.add_column("uid", justify="right", style="cyan", no_wrap=True)
            table.add_column("weight", style="magenta")
            for index, weight in list(zip(ui.tolist(), ws.tolist())):
                table.add_row(str(index), str(round(weight, 4)))
            console = Console()
            console.print(table)

        try:
            bt.logging.debug("Setting weights.")
            await asyncio.wait_for(_try_set_weights(), ttl)
            bt.logging.debug("Finished setting weights.")
        except asyncio.TimeoutError:
            bt.logging.error(f"Failed to set weights after {ttl} seconds")

    async def try_sync_metagraph(self, ttl: int):
        def sync_metagraph(endpoint):
            # Update self.metagraph
            self.metagraph = bt.subtensor(endpoint).metagraph(self.config.netuid)
            self.metagraph.save()

        process = multiprocessing.Process(
            target=sync_metagraph, args=(self.subtensor.chain_endpoint,)
        )
        process.start()
        process.join(timeout=ttl)
        if process.is_alive():
            process.terminate()
            process.join()
            bt.logging.error(f"Failed to sync metagraph after {ttl} seconds")
            return

        bt.logging.info("Synced metagraph")
        self.metagraph.load()
        self.miner_iterator.set_miner_uids(self.metagraph.uids.tolist())

    async def try_run_step(self, ttl: int):
        async def _try_run_step():
            await self.run_step()

        try:
            bt.logging.trace("Running step.")
            await asyncio.wait_for(_try_run_step(), ttl)
            bt.logging.trace("Finished running step.")
        except asyncio.TimeoutError:
            bt.logging.error(f"Failed to run step after {ttl} seconds")

    async def run_step(self):
        """
        Executes a step in the evaluation process of models. This function performs several key tasks:
        1. Identifies valid models for evaluation (top sample_min from last run + newly updated models).
        2. Generates random pages for evaluation and prepares batches for each page from the dataset.
        3. Computes the scoring for each model based on the losses incurred on the evaluation batches.
        4. Calculates wins and win rates for each model to determine their performance relative to others.
        5. Updates the weights of each model based on their performance and applies a softmax normalization.
        6. Implements a blacklist mechanism to remove underperforming models from the evaluation set.
        7. Logs all relevant data for the step, including model IDs, pages, batches, wins, win rates, and losses.
        """

        # Update self.metagraph
        await self.try_sync_metagraph(ttl=60)

        competition_parameters = constants.COMPETITION_SCHEDULE[
            self.global_step % len(constants.COMPETITION_SCHEDULE)
        ]

        # Pull relevant uids for step. If they aren't found in the model tracker on eval they will be skipped.
        uids = list(self.uids_to_eval[competition_parameters.competition_id])

        if not uids:
            if self.config.genesis:
                bt.logging.debug(
                    f"No uids to eval for competition {competition_parameters.competition_id}. Waiting 5 minutes to download some models."
                )
                time.sleep(300)
            else:
                bt.logging.debug(
                    f"No uids to eval for competition {competition_parameters.competition_id}."
                )
            return

        # Prepare evaluation
        bt.logging.debug(
            f"Computing metrics on {uids} for competition {competition_parameters.competition_id}"
        )
        bt.logging.info("run_step_count = {},  uids_queue.epochs = {}".format(self.run_step_count, self.uids_queue.epochs))
        scores_per_uid = {muid: None for muid in uids}
        sample_per_uid = {muid: None for muid in uids}

        load_model_perf = PerfMonitor("Eval: Load model")
        compute_loss_perf = PerfMonitor("Eval: Compute loss")

        self.model_tracker.release_all()
        is_duplicate = []
        not_in_tracker = []
        uid_to_hotkey_and_model_metadata: typing.Dict[
            int, typing.Tuple[str, typing.Optional[ModelMetadata]]
        ] = {}
        uid_to_seed = {uid: seed for uid, seed in zip(uids, self.seeds_to_eval)}
        for uid_i in uids:
            # Check that the model is in the tracker.
            hotkey = self.metagraph.hotkeys[uid_i]

            if self.uids_queue.epochs == 0 and uid_i not in self.uid_last_checked:
                # On the first step, don't ignore models that have not been synced unless we have tried to sync and failed.
                hotkey = self.metagraph.hotkeys[uid_i]
                while True:
                    # Retry if we get a timeout while querying the chain for model metadata.
                    try:
                        asyncio.run(self.model_updater.sync_model_metadata_only(hotkey))
                        break
                    except TimeoutError as e:
                        pass

            model_i_metadata = self.model_tracker.take_model_metadata_for_miner_hotkey(
                hotkey
            )

            if model_i_metadata is not None:
                for other_uid, (
                    other_hotkey,
                    other_metadata,
                ) in uid_to_hotkey_and_model_metadata.items():
                    if (
                        other_metadata
                        and model_i_metadata.id.hash == other_metadata.id.hash
                    ):
                        if model_i_metadata.block < other_metadata.block:
                            bt.logging.debug(f"Preferring duplicate of {uid_i} over {other_uid} since it is older")
                            # Release the other model since it is not in use.
                            self.model_tracker.release_model_metadata_for_miner_hotkey(other_hotkey, other_metadata)
                            uid_to_hotkey_and_model_metadata[other_uid] = (
                                other_hotkey,
                                None,
                            )
                            is_duplicate.append(other_uid)
                        else:
                            bt.logging.debug(f"Preferring duplicate of {other_uid} over {uid_i} since it is older")
                            # Release own model since it is not in use.
                            self.model_tracker.release_model_metadata_for_miner_hotkey(hotkey, model_i_metadata)
                            model_i_metadata = None
                            is_duplicate.append(uid_i)
                        break
            else:
                is_vali = self.metagraph.Tv[uid_i] > 0
                if is_vali:
                    bt.logging.debug(
                        f"Model for {uid_i} is a validator and not in the tracker. Skipping."
                    )
                else:
                    bt.logging.debug(
                        f"Model for {uid_i} is not in the tracker. Skipping."
                    )
                not_in_tracker.append(uid_i)

            uid_to_hotkey_and_model_metadata[uid_i] = (hotkey, model_i_metadata)

        seed = random.randint(0, 2**16)

        batch_size = self.config.batch_size
        n_samples = self.config.num_samples_per_eval
        n_bootstrap = 256
        pbar = tqdm.tqdm(uid_to_hotkey_and_model_metadata.items(), desc="Rate Models", leave=True)
        n_eval_total = len(uid_to_hotkey_and_model_metadata)

        for idx, (uid_i, (
            hotkey,
            model_i_metadata,
        )) in enumerate(pbar):
            scores = None

            if model_i_metadata is not None:
                if (
                    model_i_metadata.id.competition_id
                    == competition_parameters.competition_id
                ):
                    if self.uids_queue.epochs == 0:
                        # It is possible that we did not download a model. We should ensure that it is downloaded.
                        asyncio.run(self.model_updater.ensure_model_downloaded(hotkey))

                    self.model_tracker.touch_miner_model(hotkey)

                    try:
                        # Get the model locally and evaluate its loss.
                        model_i = None
                        with load_model_perf.sample():
                            model_i = self.local_store.retrieve_model(
                                hotkey, model_i_metadata.id, competition_parameters
                            )

                        with compute_loss_perf.sample():
                            bt.logging.info(
                                f"{idx}/{n_eval_total} Computing loss for uid: {uid_i}, ckpt: {model_i.ckpt}"
                            )
                            seed = uid_to_seed[uid_i]
                            with threadpool_limits(limits=1, user_api="blas"):
                                scores = rate(
                                    model_i.ckpt,
                                    competition_parameters.competition_id,
                                    seed,
                                    samples=n_samples,
                                    n_bootstrap=256,
                                    batch_size=batch_size,
                                )

                        del model_i
                        torch.cuda.empty_cache()
                    except Exception as e:
                        bt.logging.error(
                            f"Error in eval loop: {e}. Setting scores for uid: {uid_i} to zero."
                        )
                    finally:
                        # After we are done with the model, release it.
                        self.model_tracker.release_model_metadata_for_miner_hotkey(hotkey, model_i_metadata)

                    # Update the block this uid last updated their model if successful.
                    self.block[uid_i] = model_i_metadata.block
                else:
                    bt.logging.debug(
                        f"Skipping {uid_i}, submission is for a different competition ({model_i_metadata.id.competition_id}). Setting score to zero."
                    )
            else:
                if uid_i in is_duplicate:
                    bt.logging.debug(
                        f"Unable to load the model for {uid_i} because duplicate. Setting score to zero."
                    )
                elif uid_i in not_in_tracker:
                    bt.logging.debug(
                        f"Unable to load the model for {uid_i} because it doesn't have a model in the tracker."
                    )
                else:
                    bt.logging.warning(f"Unable to load model for {uid_i}")

            scores_per_uid[uid_i] = scores
            if scores is not None:
                bt.logging.debug(f"Computed model scores for uid {uid_i}. Mean: {np.array(scores).mean()}")

        # Update the first and second moments.
        # Use a exponential moving average to update the sample mean and variance.
        ema_alpha = self.ema_alpha
        for uid_i in uids:
            sample_mean_prev, sample_var_prev = self.sample_mean_per_uid[uid_i], self.sample_var_per_uid[uid_i]
            scores = scores_per_uid[uid_i]

            if scores is None:
                scores = [0.0 for _ in range(n_bootstrap)]
            else:
                self.count_per_uid[uid_i] += 1

            sample_mean = np.mean(scores)
            sample_var = np.var(scores, ddof=1)

            self.sample_mean_per_uid[uid_i] = ema_alpha * sample_mean_prev + (1-ema_alpha) * sample_mean
            self.sample_var_per_uid[uid_i] = ema_alpha * sample_var_prev + (1-ema_alpha) * sample_var

        # Compute wins and win rates per uid.
        num = 8
        sample_mean_per_uid_corrected, sample_var_per_uid_corrected = self.sample_stats_corrected
        win_rate = compute_wins(sample_mean_per_uid_corrected, sample_var_per_uid_corrected, self.block, num=num)

        if self.uids_queue.epoch_is_done:
            # Make sure the winrate of any uids that we have never evaluated is zero, so we don't give validators
            # positive winrate.
            win_rate[self.count_per_uid == 0] = 0

        # Compute softmaxed weights based on win rate.
        model_weights = torch.tensor(win_rate, dtype=torch.float32)
        new_weights = torch.softmax(model_weights / constants.temperature, dim=0)

        # Update weights based on moving average.
        scale = (
            len(constants.COMPETITION_SCHEDULE)
            * competition_parameters.reward_percentage
        )
        new_weights *= scale / new_weights.sum()
        if new_weights.shape[0] < self.weights.shape[0]:
            self.weights = self.weights[: new_weights.shape[0]]
        elif new_weights.shape[0] > self.weights.shape[0]:
            self.weights = torch.cat(
                [
                    self.weights,
                    torch.zeros(new_weights.shape[0] - self.weights.shape[0]),
                ]
            )
        alpha = constants.alpha

        # Only update the weights once we have completed a full epoch, for each epoch.
        if self.uids_queue.epoch_is_done:
            self.weights = (
                alpha * self.weights + (1 - alpha) * new_weights
            )
        self.weights = self.weights.nan_to_num(0.0)

        # Randomly sample a new set of uids from the list of models that we currently have.
        self.seeds_to_eval, uids_to_eval = self.uids_queue.take(16)
        self.uids_to_eval[competition_parameters.competition_id] = uids_to_eval

        # Log the performance of the eval loop.
        bt.logging.debug(load_model_perf.summary_str())
        bt.logging.debug(compute_loss_perf.summary_str())

        # Log to screen and wandb.
        scores_per_uid = {k: v for k, v in scores_per_uid.items() if v is not None}
        uids = list(scores_per_uid.keys())

        try:
            self.log_step(
                competition_parameters.competition_id,
                uids,
                win_rate,
                scores_per_uid,
                sample_per_uid,
                load_model_perf.summary_str(),
                compute_loss_perf.summary_str(),
            )
        except Exception as e:
            bt.logging.warning(f"Error during log_step: {e}")

        # Increment the number of completed run steps by 1
        self.run_step_count += 1

    def log_step(
        self,
        competition_id,
        uids,
        win_rate,
        losses_per_uid,
        sample_per_uid,
        load_model_perf_str,
        compute_loss_perf_str,
    ):
        # Build step log
        step_log = {
            "timestamp": time.time(),
            "competition_id": competition_id,
            "uids": uids,
            "uid_data": {},
        }
        for i, uid in enumerate(uids):
            step_log["uid_data"][str(uid)] = {
                "uid": uid,
                "block": self.block[uid],
                "average_loss": (
                    sum(losses_per_uid[uid]) / len(losses_per_uid[uid])
                    if len(losses_per_uid[uid]) > 0
                    else math.inf
                ),
                "win_rate": win_rate[uid],
                "weight": self.weights[uid].item(),
                "sample_prompt": (
                    sample_per_uid[uid][0] if sample_per_uid[uid] is not None else None
                ),
                "sample_response": (
                    sample_per_uid[uid][1] if sample_per_uid[uid] is not None else None
                ),
                "sample_truth": (
                    sample_per_uid[uid][2] if sample_per_uid[uid] is not None else None
                ),
            }
        table = Table(title="Step")
        table.add_column("uid", justify="right", style="cyan", no_wrap=True)
        table.add_column("average_loss", style="magenta")
        table.add_column("win_rate", style="magenta")
        table.add_column("weights", style="magenta")
        table.add_column("block", style="magenta")
        for uid in uids:
            try:
                table.add_row(
                    str(uid),
                    str(round(step_log["uid_data"][str(uid)]["average_loss"], 4)),
                    str(round(step_log["uid_data"][str(uid)]["win_rate"], 4)),
                    str(round(self.weights[uid].item(), 4)),
                    str(step_log["uid_data"][str(uid)]["block"]),
                )
            except:
                pass
        console = Console()
        console.print(table)

        ws, ui = self.weights.topk(len(self.weights))
        table = Table(title="Weights > 0.001")
        table.add_column("uid", justify="right", style="cyan", no_wrap=True)
        table.add_column("weight", style="magenta")
        for index, weight in list(zip(ui.tolist(), ws.tolist())):
            if weight > 0.001:
                table.add_row(str(index), str(round(weight, 4)))
        console = Console()
        console.print(table)

        # Sink step log.
        bt.logging.trace(f"Step results: {step_log}")

        if self.config.wandb_project and not self.config.offline:
            # If we have already completed X steps then we will complete the current wandb run and make a new one.
            if (
                self.config.wandb_max_steps_per_run
                and self.run_step_count
                and self.run_step_count % self.config.wandb_max_steps_per_run == 0
            ):
                bt.logging.trace(
                    f"Validator has completed {self.run_step_count} run steps. Creating a new wandb run."
                )
                self.wandb_run.finish()
                self.new_wandb_run()

            original_format_json = json.dumps(step_log)
            uids = step_log["uids"]
            uid_data = step_log["uid_data"]

            # Create a new dictionary with the required format
            graphed_data = {
                "time": time.time(),
                "competition_id": competition_id,
                "block": self.metagraph.block.item(),
                "uid_data": {
                    str(uid): uid_data[str(uid)]["average_loss"] for uid in uids
                },
                "win_rate_data": {
                    str(uid): uid_data[str(uid)]["win_rate"] for uid in uids
                },
                "sample_prompt_data": {
                    str(uid): uid_data[str(uid)]["sample_prompt"] for uid in uids
                },
                "sample_response_data": {
                    str(uid): uid_data[str(uid)]["sample_response"] for uid in uids
                },
                "sample_truth_data": {
                    str(uid): uid_data[str(uid)]["sample_truth"] for uid in uids
                },
                "weight_data": {str(uid): self.weights[uid].item() for uid in uids},
                "load_model_perf_log": load_model_perf_str,
                "compute_model_perf_log": compute_loss_perf_str,
            }
            bt.logging.trace("Logging to Wandb")
            self.wandb_run.log(
                {**graphed_data, "original_format_json": original_format_json},
                step=self.global_step,
            )

    async def run(self):
        while True:
            try:
                while (
                    self.metagraph.block.item() - self.last_epoch
                    < self.config.blocks_per_epoch
                ):
                    await self.try_run_step(ttl=60 * 40)
                    bt.logging.debug(
                        f"{self.metagraph.block.item() - self.last_epoch } / {self.config.blocks_per_epoch} blocks until next epoch."
                    )
                    self.global_step += 1

                if not self.config.dont_set_weights and not self.config.offline:
                    await self.try_set_weights(ttl=120)
                self.last_epoch = self.metagraph.block.item()
                self.epoch_step += 1

            except KeyboardInterrupt:
                bt.logging.info(
                    "KeyboardInterrupt caught, gracefully closing the wandb run..."
                )
                if self.config.wandb_project and not self.config.offline:
                    self.wandb_run.finish()
                exit()

            except Exception as e:
                bt.logging.error(
                    f"Error in validator loop \n {e} \n {traceback.format_exc()}"
                )


if __name__ == "__main__":
    asyncio.run(Validator().run())
