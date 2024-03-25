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
        competition_id="p225",
    ),
]
ORIGINAL_COMPETITION_ID = "p225"


assert math.isclose(sum(x.reward_percentage for x in COMPETITION_SCHEDULE), 1.0)

# ---------------------------------
# Miner/Validator Model parameters.
# ---------------------------------

weights_version_key = 1

# validator weight moving average term
alpha = 0.9
# validator scoring exponential temperature
temperature = 0.08
# validator score boosting for earlier models.
timestamp_epsilon = 0.01
