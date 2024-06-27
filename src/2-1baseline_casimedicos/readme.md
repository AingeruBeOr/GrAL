This model is finetuned by:

- Dataset: CasiMedicos (english)
- Architecture: Baseline 

Executable files:
- `train.py`: to finetune the model. Outputs:
    - `training_arguments.json`: the training arguments used for the finetuning.
    - ``
- `hyperp_sweep.py`: for hyperparameter sweep.

Axuiliary files:
- `ModelDataTrainingArguments.py`: 
- `eval_function`: evaluation function to use during training.