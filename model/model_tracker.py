import copy
import datetime
import threading
from typing import Dict, List, Optional, Set
import pickle
import bittensor as bt
from model.data import ModelMetadata


class ModelTracker:
    """Tracks the current model for each miner.

    Thread safe.
    """

    def __init__(
        self,
    ):
        # Create a dict from miner hotkey to model metadata.
        self.miner_hotkey_to_model_metadata_dict: dict[str, ModelMetadata] = dict()
        # Create a dict from miner hotkey to last time it was evaluated/loaded/updated
        self.miner_hotkey_to_last_touched_dict: dict[str, datetime.datetime] = dict()

        # List of overwritten models that may be safe to delete if not curently in use.
        self.old_model_metadata: list[tuple[str, ModelMetadata]] = []
        # List of model metadata that are currently in use.
        self.model_metadata_in_use: set[tuple[str, str]] = set()

        # Make this class thread safe because it will be accessed by multiple threads.
        # One for the downloading new models loop and one for the validating models loop.
        self.lock = threading.RLock()

    def save_state(self, filepath):
        """Save the current state to the provided filepath."""

        # Open a writable binary file for pickle.
        with self.lock:
            with open(filepath, "wb") as f:
                pickle.dump(self.miner_hotkey_to_model_metadata_dict, f)

    def load_state(self, filepath):
        """Load the state from the provided filepath."""

        # Open a readable binary file for pickle.
        with open(filepath, "rb") as f:
            self.miner_hotkey_to_model_metadata_dict = pickle.load(f)

    def get_miner_hotkey_to_model_metadata_dict(self) -> Dict[str, ModelMetadata]:
        """Returns the mapping from miner hotkey to model metadata."""

        # Return a copy to ensure outside code can't modify the scores.
        with self.lock:
            return copy.deepcopy(self.miner_hotkey_to_model_metadata_dict)

    def get_model_metadata_for_miner_hotkey(
        self, hotkey: str
    ) -> Optional[ModelMetadata]:
        """Returns the model metadata for a given hotkey if any."""

        with self.lock:
            if hotkey in self.miner_hotkey_to_model_metadata_dict:
                return self.miner_hotkey_to_model_metadata_dict[hotkey]
            return None

    def take_model_metadata_for_miner_hotkey(self, hotkey: str) -> Optional[ModelMetadata]:
        """Returns the model metadata for a given hotkey if any. Also, marks it as in use to prevent race conditions."""

        with self.lock:
            if hotkey in self.miner_hotkey_to_model_metadata_dict:
                metadata = self.miner_hotkey_to_model_metadata_dict[hotkey]
                self.model_metadata_in_use.add((hotkey, metadata.id.hash))
                return metadata
            return None

    def release_all(self):
        with self.lock:
            self.model_metadata_in_use.clear()

    def release_model_metadata_for_miner_hotkey(self, hotkey: str, metadata: ModelMetadata):
        with self.lock:
            pair = (hotkey, metadata.id.hash)
            if pair not in self.model_metadata_in_use:
                bt.logging.error("Model metadata is not in use!")

            if (hotkey, metadata) in self.old_model_metadata:
                bt.logging.trace("Releasing old model metadata for hotkey: {}", hotkey)

            self.model_metadata_in_use.remove(pair)

    def get_miner_hotkey_to_last_touched_dict(self) -> Dict[str, datetime.datetime]:
        """Returns the mapping from miner hotkey to last time it was touched."""

        # Return a copy to ensure outside code can't modify the scores.
        with self.lock:
            return copy.deepcopy(self.miner_hotkey_to_last_touched_dict)

    def on_hotkeys_updated(self, incoming_hotkeys: Set[str]):
        """Notifies the tracker which hotkeys are currently being tracked on the metagraph."""

        with self.lock:
            existing_hotkeys = set(self.miner_hotkey_to_model_metadata_dict.keys())
            for hotkey in existing_hotkeys - incoming_hotkeys:
                del self.miner_hotkey_to_model_metadata_dict[hotkey]
                bt.logging.trace(f"Removed outdated hotkey metadata: {hotkey} from ModelTracker")

            existing_hotkeys = set(self.miner_hotkey_to_last_touched_dict.keys())
            for hotkey in existing_hotkeys - incoming_hotkeys:
                del self.miner_hotkey_to_last_touched_dict[hotkey]
                bt.logging.trace(f"Removed outdated hotkey timestamp: {hotkey} from ModelTracker")

    def get_and_clear_old_models(self) -> list[tuple[str, ModelMetadata]]:
        with self.lock:
            to_delete = []
            still_in_use = []
            for hotkey, model in self.old_model_metadata:
                if (hotkey, model.id.hash) in self.model_metadata_in_use:
                    still_in_use.append((hotkey, model))
                else:
                    to_delete.append((hotkey, model))
            self.old_model_metadata = still_in_use

        return to_delete

    def on_miner_model_updated(
        self,
        hotkey: str,
        model_metadata: ModelMetadata,
    ) -> None:
        """Notifies the tracker that a miner has had their associated model updated.

        Args:
            hotkey (str): The miner's hotkey.
            model_metadata (ModelMetadata): The latest model metadata of the miner.
        """
        with self.lock:
            if hotkey in self.miner_hotkey_to_model_metadata_dict:
                old_metadata = self.miner_hotkey_to_model_metadata_dict[hotkey]
                self.old_model_metadata.append((hotkey, old_metadata))

            self.miner_hotkey_to_model_metadata_dict[hotkey] = model_metadata
            self.miner_hotkey_to_last_touched_dict[hotkey] = datetime.datetime.now()

            bt.logging.trace(f"Updated Miner {hotkey}. ModelMetadata={model_metadata}.")

    def touch_miner_model(self, hotkey: str) -> None:
        """Notifies the tracker that a miner has been touched."""

        now = datetime.datetime.now()
        with self.lock:
            self.miner_hotkey_to_last_touched_dict[hotkey] = now

            bt.logging.trace(f"Touched Miner {hotkey}. datetime={now}.")

    def touch_all_miner_models(self) -> None:
        """Touch all miner models."""

        now = datetime.datetime.now()
        with self.lock:
            for hotkey in list(self.miner_hotkey_to_model_metadata_dict.keys()):
                self.miner_hotkey_to_last_touched_dict[hotkey] = now

            bt.logging.trace(f"Touched All Miners. datetime={now}.")
