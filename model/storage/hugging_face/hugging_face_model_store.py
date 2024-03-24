import tempfile
import os
from huggingface_hub import HfApi
from model.data import Model, ModelId
from model.storage.disk import utils
from transformers import AutoModelForCausalLM, AutoTokenizer
from constants import CompetitionParameters, MAX_HUGGING_FACE_BYTES

from model.storage.remote_model_store import RemoteModelStore
import constants
from huggingface_hub import HfApi
from safetensors.torch import load_file, save_file
from collections import defaultdict
import torch


def shared_pointers(tensors):
    ptrs = defaultdict(list)
    for k, v in tensors.items():
        ptrs[v.data_ptr()].append(k)
    failing = []
    for ptr, names in ptrs.items():
        if len(names) > 1:
            failing.append(names)
    return failing


class HuggingFaceModelStore(RemoteModelStore):
    """Hugging Face based implementation for storing and retrieving a model."""

    @classmethod
    def assert_access_token_exists(cls) -> str:
        """Asserts that the access token exists."""
        if not os.getenv("HF_ACCESS_TOKEN"):
            raise ValueError("No Hugging Face access token found to write to the hub.")
        return os.getenv("HF_ACCESS_TOKEN")

    async def upload_model(
        self, model: Model, competition_parameters: CompetitionParameters
    ) -> ModelId:
        """Uploads a trained model to Hugging Face."""
        token = HuggingFaceModelStore.assert_access_token_exists()
        api = HfApi(token=token)
        api.create_repo(
            repo_id=model.id.namespace + "/" + model.id.name,
            exist_ok=True,
            private=True,
        )
        # convert to safetensors if needed
        if not model.ckpt.endswith(".safetensors"):
            # save the model as a safetensors file
            loaded = torch.load(model.ckpt, map_location="cpu")
            if "model" in loaded:
                loaded = loaded["model"]
            shared = shared_pointers(loaded)
            for shared_weights in shared:
                for name in shared_weights[1:]:
                    loaded.pop(name)

            # For tensors to be contiguous
            loaded = {k: v.contiguous() for k, v in loaded.items()}
            save_file(loaded, model.ckpt + ".safetensors", metadata={"format": "pt"})
            model.ckpt = model.ckpt + ".safetensors"

        commit_info = api.upload_file(
            path_or_fileobj=model.ckpt,
            path_in_repo="checkpoint.safetensors",
            repo_id=model.id.namespace + "/" + model.id.name,
        )
        model_id_with_commit = ModelId(
            namespace=model.id.namespace,
            name=model.id.name,
            hash=model.id.hash,
            commit=commit_info.oid,
            competition_id=model.id.competition_id,
        )

        # TODO consider skipping the redownload if a hash is already provided.
        # To get the hash we need to redownload it at a local tmp directory after which it can be deleted.
        with tempfile.TemporaryDirectory() as temp_dir:
            model_with_hash = await self.download_model(
                model_id_with_commit, temp_dir, competition_parameters
            )
            # Return a ModelId with both the correct commit and hash.
            return model_with_hash.id

    async def download_model(
        self,
        model_id: ModelId,
        local_path: str,
        model_parameters: CompetitionParameters,
    ) -> Model:
        """Retrieves a trained model from Hugging Face."""
        if not model_id.commit:
            raise ValueError("No Hugging Face commit id found to read from the hub.")

        repo_id = model_id.namespace + "/" + model_id.name

        # Check ModelInfo for the size of model.safetensors file before downloading.
        try:
            token = HuggingFaceModelStore.assert_access_token_exists()
        except:
            token = None
        api = HfApi(token=token)
        model_info = api.model_info(
            repo_id=repo_id, revision=model_id.commit, timeout=10, files_metadata=True
        )
        size = sum(repo_file.size for repo_file in model_info.siblings)
        if size > MAX_HUGGING_FACE_BYTES:
            raise ValueError(
                f"Hugging Face repo over maximum size limit. Size {size}. Limit {MAX_HUGGING_FACE_BYTES}."
            )

        api.hf_hub_download(
            repo_id=repo_id,
            revision=model_id.commit,
            filename="checkpoint.safetensors",
            cache_dir=local_path,
        )

        # Get the directory the model was stored to.
        model_dir = utils.get_hf_download_path(local_path, model_id)

        # Realize all symlinks in that directory since Transformers library does not support avoiding symlinks.
        utils.realize_symlinks_in_directory(model_dir)

        # Compute the hash of the downloaded model.
        model_hash = utils.get_hash_of_directory(model_dir)
        model_id_with_hash = ModelId(
            namespace=model_id.namespace,
            name=model_id.name,
            commit=model_id.commit,
            hash=model_hash,
            competition_id=model_id.competition_id,
        )

        return Model(id=model_id_with_hash, ckpt=model_dir)
