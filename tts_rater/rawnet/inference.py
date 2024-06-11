from .rawnet3 import RawNet3
from .rawnetblock import Bottle2neck
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

class AntiSpoofingInference:
    def __init__(self, device="cuda"):
        self.model = RawNet3(
                Bottle2neck,
                model_scale=8,
                context=True,
                summed=True,
                encoder_type="ECA",
                nOut=256,
                out_bn=False,
                sinc_stride=10,
                log_sinc=True,
                norm_sinc="mean",
                grad_mult=1,
            )
        if device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        model_name = 'jungjee/RawNet3'
        model_path = model_name.replace('/', '_')
        temp_location = hf_hub_download(repo_id=model_name, repo_type='model', filename='model.pt', local_dir=model_path)
        self.model.load_state_dict(torch.load(temp_location)['model'])
        self.model.eval()
        self.model.to(self.device)
    
    def get_embedding(self, x):
        with torch.inference_mode():
            return self.model(x)
