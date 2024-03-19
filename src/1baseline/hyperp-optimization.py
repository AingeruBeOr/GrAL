import sys
import os
import json
import wandb
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import evaluate
import numpy as np
from variables import PATH_TO_MEDMCQA_TRAIN, PATH_TO_MEDMCQA_DEV, PATH_TO_MEDMCQA_LOADER, PATH_TO_ERIBERTA


def optuna_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "gradient_accumulation_steps": trial.suggest_int("gradient_accumulation_steps", 1, 50),
        "epochs": trial.suggest_int("epochs", 1, 10),
    }


def optimization_objective(metrics):
    return metrics[training_args.metric_for_best_model]


## 0. SET UP ##
# Load hyperparameters from json file
if len(sys.argv) == 2:
    with open(sys.argv[1], 'r') as file:
        training_arguments = json.load(file)
else:
    raise ValueError("Usage: python train.py <path_to_json_file>")

# Set up Weights & Biases
os.environ["WANDB_PROJECT"] = "tfg-baseline" # set the wandb project where this run will be logged
os.environ["WANDB_LOG_MODEL"] = "false" # do not upload the model to wandb


## 1. LOAD DATASET ##
dataset = load_dataset(path = PATH_TO_MEDMCQA_LOADER, 
                       data_files={"train": PATH_TO_MEDMCQA_TRAIN, 
                                   "dev": PATH_TO_MEDMCQA_DEV})

train_shuffled = dataset['train'].shuffle(seed=42)
dataset['train'] = train_shuffled.select(range(10000)) # select only the first 10000 instances

print(f'Dataset length to hyperparameter search: {len(dataset["train"])}')

## 2. TOKENIZATION ##
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(PATH_TO_ERIBERTA, use_fast=True)

# Preprecess function
def preprocess_function(example):
    my_input = f'{example["question"]}</s>{example["opa"]}</s>{example["opb"]}</s>{example["opc"]}</s>{example["opd"]}'
    return tokenizer(my_input, padding='max_length', max_length=512)

# Tokenize and encode
dataset = dataset.map(preprocess_function)

## 3. FINE-TUNE MODEL ##
# Load model. Transformer with default classification head
def model_init():
    return AutoModelForSequenceClassification.from_pretrained(PATH_TO_ERIBERTA, num_labels=4)

# Set evaluetaion function to be used during training (accuracy, f1, ...)
accuracy = evaluate.load('accuracy') # Load the accuracy function
f1 = evaluate.load('f1') # Load the f-score function
precision = evaluate.load('precision') # Load the precision function
recall = evaluate.load('recall') # Load the recall function
cnf_matrix = evaluate.load('BucketHeadP65/confusion_matrix') # Load the confusion matrix function

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy_value = accuracy.compute(predictions=predictions, references=labels)

    f1_value_micro = f1.compute(predictions=predictions, references=labels, average='micro')
    f1_value_macro = f1.compute(predictions=predictions, references=labels, average='macro')
    f1_value_weighted = f1.compute(predictions=predictions, references=labels, average='weighted')
    class_f1 = f1.compute(predictions=predictions, references=labels, average=None)
    
    precision_value_micro = precision.compute(predictions=predictions, references=labels, average='micro')
    precision_value_macro = precision.compute(predictions=predictions, references=labels, average='macro')
    precision_value_weighted = precision.compute(predictions=predictions, references=labels, average='weighted')
    class_precision = precision.compute(predictions=predictions, references=labels, average=None, zero_division='warn')
    
    recall_value_micro = recall.compute(predictions=predictions, references=labels, average='micro')
    recall_value_macro = recall.compute(predictions=predictions, references=labels, average='macro')
    recall_value_weighted = recall.compute(predictions=predictions, references=labels, average='weighted')
    class_recall = recall.compute(predictions=predictions, references=labels, average=None, zero_division='warn')

    confusion_matrix_serializable = cnf_matrix.compute(predictions=predictions, references=labels)
    
    # Every element in the return dict, must be serializable so we log confusion_matrix (which is a plot object from wanbd library) independently to wandb  
    confusion_matrix = wandb.plot.confusion_matrix(probs=None, y_true=labels, preds=predictions, class_names=['A', 'B', 'C', 'D'])
    wandb.log({'confusion_matrix': confusion_matrix})

    return {
        'accuracy': accuracy_value['accuracy'],
        'f1_micro': f1_value_micro['f1'],
        'f1_macro': f1_value_macro['f1'],
        'f1_weighted': f1_value_weighted['f1'],
        'f1_opa': class_f1['f1'][0],
        'f1_opb': class_f1['f1'][1],
        'f1_opc': class_f1['f1'][2],
        'f1_opd': class_f1['f1'][3],
        'precision_micro': precision_value_micro['precision'],
        'precision_macro': precision_value_macro['precision'],
        'precision_weighted': precision_value_weighted['precision'],
        'precision_opa': class_precision['precision'][0],
        'precision_opb': class_precision['precision'][1],
        'precision_opc': class_precision['precision'][2],
        'precision_opd': class_precision['precision'][3],
        'recall_micro': recall_value_micro['recall'],
        'recall_macro': recall_value_macro['recall'],
        'recall_weighted': recall_value_weighted['recall'],
        'recall_opa': class_recall['recall'][0],
        'recall_opb': class_recall['recall'][1],
        'recall_opc': class_recall['recall'][2],
        'recall_opd': class_recall['recall'][3],
        'confusion_matrix': confusion_matrix_serializable['confusion_matrix'].tolist() # convert to list to be serializable
    }

# SET TRAINING HYPERPARAMETERS #
training_args = TrainingArguments(
    output_dir=training_arguments['output_dir'],                                     # output directory. Carpeta donde se generarán todos los ficheros de salida y checkpoints. (No es necesario que la carpeta exista, el script crea todas las carpetas necesarias)
    overwrite_output_dir=training_arguments['overwrite_output_dir'],                 # overwrite the content of the output directory
    
    warmup_steps=training_arguments['warmup_steps'],                                 # number of warmup steps for learning rate scheduler
    learning_rate=training_arguments['learning_rate'],                               # learning rate
    
    num_train_epochs=training_arguments['epochs'],                                   # total number of training epochs
    per_device_train_batch_size=training_arguments['per_device_train_batch_size'],   # batch size per device during training
    per_device_eval_batch_size=training_arguments['per_device_eval_batch_size'],     # batch size for evaluation
    gradient_accumulation_steps=training_arguments['gradient_accumulation_steps'],   # gradient accumulation to avoid OOM errors
    
    evaluation_strategy=training_arguments['evaluation_strategy'],                   # when to evaluate. Options: 'no', 'steps', 'epoch'
    eval_steps=training_arguments['eval_steps'],                                     # when to evaluate (and to print log if logging_strategy = 'steps')
    
    logging_dir=training_arguments['logging_dir'],                                   # directory for storing logs
    logging_steps=training_arguments['logging_steps'],                               # when to print log (and to evaluate if evaluation_strategy = 'steps')
    report_to=training_arguments['report_to'],                                       # report to Weights & Biases
    
    metric_for_best_model=training_arguments['metric_for_best_model'],               # metric to use for best model selection
    greater_is_better=training_arguments['greater_is_better'],                       # whether the best model is the one with the highest or lowest value of the metric

    save_strategy=training_arguments['save_strategy'],                               # when to save each model checkpoint. It must be the same as 'evaluation_strategy' (porque guarda el modelo después de evaluarlo)
    save_steps=training_arguments['save_steps'],                                     # when to save each model checkpoint
    save_total_limit=training_arguments['save_total_limit'],                         # number of model checkpoints to keep
    load_best_model_at_end=training_arguments['load_best_model_at_end']              # load or not best model at the end
    
    #weight_decay=
    #use_mps_device=True                                                             # para que use la GPU del MAC
)

# SET TRAINER: MODEL TO TRAIN, TRAINING ARGUMENTS, DATSET SPLITS...
trainer = Trainer(
    model=None,
    args=training_args,              # training arguments, defined above
    train_dataset=dataset['train'],  # training dataset. Format: torch.utils.data.Dataset
    eval_dataset=dataset['dev'],     # evaluation dataset. Format: torch.utils.data.Dataset
    compute_metrics=compute_metrics, # evaluation function
    model_init=model_init            # model to train
)

# HYPERPARAMETER TUNING #
best_run = trainer.hyperparameter_search(
    backend= "optuna",                        # optimization library (backend)
    compute_objective=optimization_objective, # define the objective function (usually loss)
    direction="minimize",                     # maximize or minimize the objective function
    hp_space=optuna_hp_space,                 # define the hyperparameter space
    n_trials=20,                              # number of trials
)

print(
    "Best run:\n"
    f"id:              {best_run.run_id}\n"
    f"objective:       {best_run.objective}\n"
    f"hyperparameters: {best_run.hyperparameters}\n"
)

print(best_run)
