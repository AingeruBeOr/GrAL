This model is finetuned using:

- Dataset: CasiMedicos (english)
- Architecture: Baseline 

Executable files:
- `train.py`: to finetune the model. Output files:
    - `training_arguments.json`: the training arguments used for the finetuning.
    - `inference_eval_results.json`: inference results over the evaluation dataset.
    - `inference_test_results.json`: inference results over the test dataset.
- `hyperp_sweep.py`: for hyperparameter sweep.

Axuiliary files:
- `ModelDataTrainingArguments.py`: extra arguments for model configuration.
- `eval_function`: evaluation function computed during training.
- `hypers_example.json`: training arguments JSON example for the training.