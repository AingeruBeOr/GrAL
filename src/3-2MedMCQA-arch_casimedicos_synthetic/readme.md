# MedMCQA-architecture and CasiMedicos-dataset

This model is finetuned using:

- Dataset: CasiMedicos (english) synthetic
- Architecture: MedMCQA 

Executable files:
- `train.py`: use pytorch-lightning for training.
- `hyperparam_search.py`: launch a hyperparameter search.

Auxiliary files:
- `args.py`: training arguments dataclass.
- `dataset.py`: dataset to load.
- `model.py`: model architecture details for training.
- `sweep.yaml`: define a hyperparameter sweep with wandb

Be careful with the batch size!! An instance becomes in a 5 instance input for the model.

Given an instance:
- Input with the option A: <s>Q</s>opa</s>
- Input with the option B: <s>Q</s>opb</s>
- Input with the option C: <s>Q</s>opc</s>
- Input with the option D: <s>Q</s>opd</s>
- Input with the option E: <s>Q</s>ope</s>

## Launching a hyperparameter sweep using wandb

1. Genenarte a YAML file.
2. Create a sweep from CLI using: `wandb sweep --project <project_name> <YAML file>`. Returns a `sweep_ID`.
3. Start an agent: `wandb agent <entity>/<project_name>/<sweep_ID>` (`entity` is the account name in wandb).

# Upload local runs to WANDB

Example: `wandb sync -p tfg-baseline-casimedicos-synthetic *`