#  Arquitecturas sobre MedMCQA

## Dataset

Recogido de [https://github.com/medmcqa/medmcqa](https://github.com/medmcqa/medmcqa)

También disponible en [huggingface/medmcqa](https://huggingface.co/datasets/medmcqa)

## Model

The model to use is [EriBERTa: A Bilingual Pre-Trained Language Model for Clinical Natural Language Processing
](https://arxiv.org/abs/2306.07373)

## Estructura del proyecto

```
.
├── data
│   └── MedMCQA/
├── models
│   └── eriberta_libre/
├── readme.md
├── requirements.txt
└── src
    ├── datasetStats.ipynb
    ├── datasetStats.py
    ├── loadDataset.py
    ├── train.py
    └── variables.py
```

## Setting training arguments

Training arguments are loaded into the train script from a JSON. See example:

```json
{
    "wandb-run": "https://wandb.ai/abellido/tfg-baseline/runs/boc00c3k", // wandb run URL
    
    "output_dir": "../models/baseline/", // output directory where the model and checkpoints are stored
    "overwrite_output_dir": false, 

    "warmup_steps": 100, 
    "learning_rate": 5e-5, // This is the maximum learning rate value. In the warm up it will be adjusted. 
    
    "epochs": 5,
    "per_device_train_batch_size": 12, // Batch size for training. Lo máximo que entre en la GPU
    "per_device_eval_batch_size": 16, // Batch size for evaluation. Lo máximo que entre en la GPU. Suele entrar más porque no hay backpropagation
    "gradient_accumulation_steps": 5, // How many 'per_device_train_batch_size' accumulate before backpropagation
                                      // Real batch-size will be (per_device_train_batch_size)x(gradient_accumulation_steps) = (1 step)
    "evaluation_strategy": "steps",
    "eval_steps": 100,

    "logging_dir": "./logs",
    "logging_steps": 10,
    "report_to": "wandb",

    "metric_for_best_model": "eval_loss",
    "greater_is_better": false,

    "save_strategy": "steps",
    "save_steps": 3000, // must be multiple of 'eval_steps'
    "save_total_limit": 3,
    "load_best_model_at_end": true,

    "description": "Se lanzó sin métricas, solo se computaba el loss" // short description
}
```

`wandb-run` and `description` fields are not used as training arguments but the `.json` file is stored with the model to check how was it trained.

## Hyperparameter search

Different possible backends: optuna/ray[tune]/wandb/sigopt