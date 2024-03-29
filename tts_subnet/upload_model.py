"""A script that pushes a model from disk to the subnet for evaluation.

Usage:
    python scripts/upload_model.py --load_model_dir <path to model> --hf_repo_id my-username/my-project --wallet.name coldkey --wallet.hotkey hotkey

Prerequisites:
   1. HF_ACCESS_TOKEN is set in the environment or .env file.
   2. load_model_dir points to a directory containing a previously trained model, with relevant ckpt file named "checkpoint.pth".
   3. Your miner is registered
"""

import asyncio
import os
import argparse
import torch
import constants
from model.storage.hugging_face.hugging_face_model_store import HuggingFaceModelStore
from model.model_updater import ModelUpdater
import bittensor as bt
from utilities import utils
from model.data import Model, ModelId
from model.storage.chain.chain_model_metadata_store import ChainModelMetadataStore
from huggingface_hub import update_repo_visibility
import time

from dotenv import load_dotenv

load_dotenv()


def get_config():
    # Initialize an argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--hf_repo_id",
        type=str,
        help="The hugging face repo id, which should include the org or user and repo name. E.g. jdoe/finetuned",
    )
    parser.add_argument(
        "--load_model_dir",
        type=str,
        default=None,
        help="If provided, loads a previously trained HF model from the specified directory",
    )
    parser.add_argument(
        "--netuid",
        type=str,
        default=constants.SUBNET_UID,
        help="The subnet UID.",
    )
    parser.add_argument(
        "--competition_id",
        type=str,
        default=constants.ORIGINAL_COMPETITION_ID,
        help="competition to mine for (use --list-competitions to get all competitions)",
    )
    parser.add_argument(
        "--list_competitions", action="store_true", help="Print out all competitions"
    )

    # Include wallet and logging arguments from bittensor
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)

    # Parse the arguments and create a configuration namespace
    config = bt.config(parser)

    return config


async def main(config: bt.config):
    # Create bittensor objects.
    bt.logging(config=config)

    wallet = bt.wallet(config=config)
    subtensor = bt.subtensor(config=config)
    metagraph: bt.metagraph = subtensor.metagraph(config.netuid)

    # Make sure we're registered and have a HuggingFace token.
    utils.assert_registered(wallet, metagraph)

    # Get current model parameters
    parameters = ModelUpdater.get_competition_parameters(config.competition_id)
    if parameters is None:
        raise RuntimeError(
            f"Could not get competition parameters for block {config.competition_id}"
        )

    repo_namespace, repo_name = utils.validate_hf_repo_id(config.hf_repo_id)
    model_id = ModelId(
        namespace=repo_namespace,
        name=repo_name,
        competition_id=config.competition_id,
    )

    model = Model(id=model_id, ckpt=config.load_model_dir)
    remote_model_store = HuggingFaceModelStore()

    model_id_with_commit = await remote_model_store.upload_model(
        model=model,
        competition_parameters=parameters,
    )

    print(
        f"Model uploaded to Hugging Face with commit {model_id_with_commit.commit} and hash {model_id_with_commit.hash}"
    )

    model_metadata_store = ChainModelMetadataStore(
        subtensor=subtensor, wallet=wallet, subnet_uid=config.netuid
    )

    # We can only commit to the chain every 20 minutes, so run this in a loop, until successful.
    while True:
        try:
            update_repo_visibility(
                model_id.namespace + "/" + model_id.name,
                private=False,
                token=os.getenv("HF_ACCESS_TOKEN"),
            )
            await model_metadata_store.store_model_metadata(
                wallet.hotkey.ss58_address, model_id_with_commit
            )

            bt.logging.success("Committed model to the chain.")
            break
        except Exception as e:
            bt.logging.error(f"Failed to advertise model on the chain: {e}")
            bt.logging.error("Retrying in 120 seconds...")
            time.sleep(120)


if __name__ == "__main__":
    # Parse and print configuration
    config = get_config()
    if config.list_competitions:
        print(constants.COMPETITION_SCHEDULE)
    else:
        print(config)
        asyncio.run(main(config))
