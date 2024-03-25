# Validator 

Validators download the models from 🤗 Hugging Face for each miner based on the Bittensor chain metadata and continuously evaluate them, setting weights based on the performance of each model based on the two metrics described in the `README.md` file.

You can view the entire validation system by reading the code in `neurons/validator.py`. Pseudocode for the validation system is as follows:
```python
    weights = zeros(256)
    while True:
        # Fetch random sample of batches to evaluate models on
        batches = get_random_sample_of_batches_from_coretex_subnet()
        
        # Fetch and or update models.
        models = get_and_update_models_from_miners()

        # Compute losses for each batch and each model
        model_losses = {}
        for model in models:
            for batch in batches:
                loss = get_loss_for_model_on_batch( model, batch )
                model_losses[ model ].append( loss )

        # Compute wins for models.
        model_wins = {}
        for model_a in models:
            for model_b in models:
                for i in len( batches )
                    # Determine if better model loss with relative block number boosting.
                    if iswin( model_losses[ model_a ][ i ], model_losses[ model_b ][ i ], block_a, block_b ):
                        model_wins[ model_a ] += 1
                            
        # End epoch.
        # Weights are computed based on the ratio of wins a model attains during the epoch.
        for model_i in models:
            weights[ model_i ] += model_wins[ model_i ] / sum( model_wins.values() )
        weights = softmax( weights / temperature, dim=0 )

        # Set weights on the chain.
        set_weights( weight )
```

The behaviour of `iswin( loss_a, loss_b, block_a, block_b)` function intentionally skews the win function to reward models which have been hosted earlier such that newer models are only better than others iff their loss is `epsilon` percent lower accoring to the following function. Currently `epsilon` is set to 1% and is a hyper parameter of the mechanism

```python
def iswin( loss_a, loss_b, block_a, block_b ):
    loss_a = (1 - constants.timestamp_epsilon) * loss_a if block_a < block_b else loss_a
    loss_b = (1 - constants.timestamp_epsilon) * loss_b if block_b < block_a else loss_b
    return loss_a < loss_b
```

It is important to note that this affects the game theoretics of the incentive landscape since miners should only update their model (thus updating their timestamp to a newer date) if they have achieved an `epsilon` better performance than their previous model. This undermines the obvious optimal strategy for miners to copy the publicly available models from other miners. They **can** and should copy other miners, but they will always obtain fewer wins compared to them until they also decrease their loss by `epsilon`.

# System Requirements

Validators will need enough disk space to store the model of every miner in the subnet. Each model is limited to 512 MB, and the validator has cleanup logic to remove old models. It is recommended to have at least 256 GB of disk space.

Validators will need enough processing power to evaluate their model. Thanks to the efficiency of MeloTTS, the validator can be run on a CPU machine. We are working on a GPU version of the validator to speed up the evaluation process.

# Getting Started

## Prerequisites

1. Follow the instruction in the `README.md` file to install the package and its dependencies.

2. Make sure you've [created a Wallet](https://docs.bittensor.com/getting-started/wallets) and [registered a hotkey](https://docs.bittensor.com/subnets/register-and-participate).

3. (Optional) Run a Subtensor instance:

Your node will run better if you are connecting to a local Bittensor chain entrypoint node rather than using Opentensor's. 
We recommend running a local node as follows and passing the ```--subtensor.network local``` flag to your running miners/validators. 
To install and run a local subtensor node follow the commands below with Docker and Docker-Compose previously installed.
```bash
git clone https://github.com/opentensor/subtensor.git
cd subtensor
docker compose up --detach
```
---

# Running the Validator

## With auto-updates

We highly recommend running the validator with auto-updates. This will help ensure your validator is always running the latest release, helping to maintain a high vtrust.

Prerequisites:
1. To run with auto-update, you will need to have [pm2](https://pm2.keymetrics.io/) installed.
2. Make sure your virtual environment is activated. This is important because the auto-updater will automatically update the package dependencies with pip.
3. Make sure you're using the main branch: `git checkout main`.

To run the validator with auto-updates, use the following command:
```bash
pm2 start neurons/validator.py --name validator --interpreter python -- --wallet.name your_wallet --wallet.hotkey your_hotkey
```

## Without auto-updates
You can run validator with:
```bash
python neurons/validator.py --wallet.name your_wallet --wallet.hotkey your_hotkey
```

# Testing the Validator
You can test the validator offline by running the following command:
```bash
python neurons/validator.py --wallet.name your_wallet --wallet.hotkey your_hotkey --wandb.off --offline
```