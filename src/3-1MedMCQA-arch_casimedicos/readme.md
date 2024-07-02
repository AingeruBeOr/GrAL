# MedMCQA-architecture and CasiMedicos-dataset

This model is finetuned using:

- Dataset: CasiMedicos (english)
- Architecture: MedMCQA 

Executable files:
- `train.py`: use pytorch-lightning for training.
- `hyperparam_search.py`: launch a hyperparameter search.

Auxiliary files:
- `args.py`: training arguments dataclass.
- `dataset.py`: dataset to load.
- `model.py`: model architecture details for training.
- `sweep.yaml`: define a hyperparameter sweep with wandb

## Launching a hyperparameter sweep using wandb

1. Genenarte a YAML file
2. Create a sweep from CLI using: `wandb sweep --project <project_name> <YAML file>`. Returns a `sweep_ID`.
3. Start an agent: `wandb agent <entity>/<project_name>/<sweep_ID>` (`entity` is the account name in wandb).
