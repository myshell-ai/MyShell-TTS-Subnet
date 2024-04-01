import bittensor as bt
from bittensor.extrinsics.serving import get_metadata
import asyncio
from model.data import ModelId
from model.storage.chain.chain_model_metadata_store import ChainModelMetadataStore
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--hotkey",
    type=str,
    help="The hotkey of the model to check",
)
args = parser.parse_args()

subtensor = bt.subtensor()
subnet_uid = 3
metagraph = subtensor.metagraph(subnet_uid)

wallet = None
model_metadata_store = ChainModelMetadataStore(subtensor, subnet_uid, wallet)

model_name = asyncio.run(model_metadata_store.retrieve_model_metadata(args.hotkey))

print(model_name)