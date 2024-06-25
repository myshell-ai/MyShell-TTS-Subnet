from pathlib import Path
from dataclasses import dataclass
from typing import Type, Optional, Any, List, Tuple
import math


@dataclass
class CompetitionParameters:
    """Class defining model parameters"""

    # Reward percentage
    reward_percentage: float
    # Competition id
    competition_id: str


# ---------------------------------
# Project Constants.
# ---------------------------------

# The validator WANDB project.
WANDB_PROJECT = "myshell-tts-subnet"
# The uid for this subnet.
SUBNET_UID = 3
# The start block of this subnet
SUBNET_START_BLOCK = 2635801
# The root directory of this project.
ROOT_DIR = Path(__file__).parent.parent
# The maximum bytes for the hugging face repo
MAX_HUGGING_FACE_BYTES: int = 512 * 1024 * 1024
# Schedule of model architectures
COMPETITION_SCHEDULE: List[CompetitionParameters] = [
    CompetitionParameters(
        reward_percentage=1.0,
        competition_id="p239",
    ),
]
ORIGINAL_COMPETITION_ID = "p239"
CONSTANT_ALPHA = 0.1 # prev: 0.2
timestamp_epsilon = 0.005

assert math.isclose(sum(x.reward_percentage for x in COMPETITION_SCHEDULE), 1.0)

# ---------------------------------
# Miner/Validator Model parameters.
# ---------------------------------

weights_version_key = 4

# validator weight moving average term. alpha = 1-lr.
lr = 0.2
# validator scoring exponential temperature
temperature = 0.08
