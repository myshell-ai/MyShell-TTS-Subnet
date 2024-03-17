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
# TODO
SUBNET_UID = 1
# The start block of this subnet
# TODO
SUBNET_START_BLOCK = 2225782
# The root directory of this project.
ROOT_DIR = Path(__file__).parent.parent
# The maximum bytes for the hugging face repo
MAX_HUGGING_FACE_BYTES: int = 15 * 1024 * 1024 * 1024
# Schedule of model architectures
COMPETITION_SCHEDULE: List[CompetitionParameters] = [
    CompetitionParameters(
        reward_percentage=1.0,
        competition_id="c1",
    ),
]
ORIGINAL_COMPETITION_ID = "c1"


assert math.isclose(sum(x.reward_percentage for x in COMPETITION_SCHEDULE), 1.0)
assert all(
    len(x.competition_id) > 0 and len(x.competition_id) <= 2
    for x in COMPETITION_SCHEDULE
)

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
# validator eval sequence length.
sequence_length = 2048

# ---------------------------------
# Data generation parameters.
# ---------------------------------

OPENAI_MODEL = "gpt-3.5-turbo-0125"
OPENAI_TEMPERATURE = 0.5
NUM_SHOT = 3
