import os
import sys
import logging
import shutil
import datetime
import wandb
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, set_seed, HfArgumentParser
from eval_function import compute_metrics
from train import setup_logger
from ModelDataTrainingArguments import ModelDataTrainingArguments

'''
Performs a hyperparameter sweep using Optuna.
Search space must be defined in the `optuna_hp_space` function.
'''


def optuna_hp_space(trial):
    '''
    Optuna docs: https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/002_configurations.html#sphx-glr-tutorial-10-key-features-002-configurations-py
                 https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial.suggest_int
    '''
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "per_device_train_batch_size": trial.suggest_int("per_device_train_batch_size", 1, 18), 
        "gradient_accumulation_steps": trial.suggest_int("gradient_accumulation_steps", 1, 8), 
        "epochs": trial.suggest_int("epochs", 1, 30),
    }


def optimization_objective(metrics):
    return metrics[training_args.metric_for_best_model]


assert len(sys.argv) == 2, "Usage: python train.py <path_to_json_file>"
training_arguments_path = sys.argv[1]

logger = logging.getLogger(__name__)

parser = HfArgumentParser((ModelDataTrainingArguments, TrainingArguments))
model_data_args: ModelDataTrainingArguments
training_args: TrainingArguments
model_data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(training_arguments_path))

# Create output directory with current timestamp
current_datetime = datetime.datetime.now()
formatted_datetime = current_datetime.strftime("%Y-%m-%d-%H-%M")
experiment_name = f"{formatted_datetime}-hyp-sweep"
training_args.output_dir = training_args.output_dir + experiment_name

# Create directory and save (Copy) the training arguments json in this directory
os.makedirs(training_args.output_dir, exist_ok=False)
shutil.copyfile(training_arguments_path, training_args.output_dir + '/training_arguments.json')

# Not using Weights & Biases for hyperparameter tuning

# Set up logger
setup_logger(training_args)

# Set seed
set_seed(training_args.seed)

## 1. LOAD DATASET ##
dataset = load_dataset(
    path=model_data_args.data_loader,
    data_files={
        "train": model_data_args.train_file,
        "dev": model_data_args.validation_file,
        "test": model_data_args.test_file
    },
)

# Get the labels
label_list = dataset['train'].features['label'].names
logger.info(label_list)

# Get the number of labels
num_labels = len(label_list)
logger.info(num_labels)

# Get the label to id mapping
label_to_id = {label: i for i, label in enumerate(label_list)}
logger.info(label_to_id)

# Get the id to label mapping
id_to_label = {i: label for i, label in enumerate(label_list)}
logger.info(id_to_label)

logger.info(f"Dataset structure: \n\n{dataset}")
logger.info(f"\n\nDataset train first instance example: \n\n{dataset['train'][0]}")

## 2. TOKENIZATION ##
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_data_args.model_name_or_path, use_fast=True)

# Preprecess function
def generate_instances(example):
    return {'option_text': f'{example["opa"]}</s>{example["opb"]}</s>{example["opc"]}</s>{example["opd"]}</s>{example["ope"]}'}

def tokenize_function(examples):
    return tokenizer(
        examples['question'],
        examples['option_text'],
        truncation='only_first',
        padding='max_length' if model_data_args.pad_to_max_length else 'longest',
        max_length=512
    )

# Tokenize and encode
logger.info("\n\nTokenizing and encoding datasets...")
raw_dataset = dataset.map(generate_instances)
dataset = raw_dataset.map(tokenize_function, batched=True, load_from_cache_file=not model_data_args.overwrite_cache, )
# print(dataset['train'][0]) # print first instance tokenized and encoded
logger.info("\tTrain dataset tokenized and encoded. Number of instances: ", len(dataset['train']))
logger.info("\tDev dataset tokenized and encoded. Number of instances: ", len(dataset['dev']), "\n\n")

# Get rid of instances with more than 512 tokens (TODO: probar otras estraegias)
"""encoded_train, number_of_deleted_instances = delete_more_than_512_tokens(encoded_train)
logger.info("\tDeleted instances with more than 512 tokens from train dataset. Number of instances: ", len(encoded_train['input_ids']), "Deleted: ", number_of_deleted_instances)
encoded_dev, number_of_deleted_instances = delete_more_than_512_tokens(encoded_dev)
logger.info("\tDeleted instances with more than 512 tokens from train dataset. Number of instances: ", len(encoded_dev['input_ids']), "Deleted: ", number_of_deleted_instances)
"""

## 3. FINE-TUNE MODEL ##
# Load model. Transformer with default classification head
def model_init(trial):
    return AutoModelForSequenceClassification.from_pretrained(
        model_data_args.model_name_or_path,
        num_labels=num_labels,
        id2label=id_to_label,
        label2id=label_to_id,
    )

# SET TRAINER: MODEL TO TRAIN, TRAINING ARGUMENTS, DATSET SPLITS...
trainer = Trainer(
    model=None,
    model_init=model_init,           # necessary when hyperparameter tuning
    tokenizer=tokenizer,
    args=training_args,              # training arguments, defined above
    train_dataset=dataset['train'],  # training dataset. Format: torch.utils.data.Dataset
    eval_dataset=dataset['dev'],     # evaluation dataset. Format: torch.utils.data.Dataset
    compute_metrics=compute_metrics  # evaluation function
)

# HYPERPARAMETER TUNING #
best_run = trainer.hyperparameter_search(
    backend="optuna",                         # optimization library (backend)
    compute_objective=optimization_objective, # define the objective function (usually loss)
    direction="maximize",                     # maximize or minimize the objective function (compute_objective function return value)
    hp_space=optuna_hp_space,                 # define the hyperparameter space
    n_trials=100,                              # number of trials. Each trial is a set of hyperparameters
)

print(
    "Best run:\n"
    f"id:              {best_run.run_id}\n"
    f"objective:       {best_run.objective}\n"
    f"hyperparameters: {best_run.hyperparameters}\n"
)

# Save a file with the best hyperparameters
with open(training_args.output_dir + '/best_hyperparameters.json', 'w') as f:
    f.write(str(best_run))
