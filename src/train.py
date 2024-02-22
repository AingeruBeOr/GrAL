from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from loadDataset import MedMCQA_Dataset, load_dataset_as_dict
from variables import PATH_TO_MEDMCQA_TRAIN, PATH_TO_MEDMCQA_DEV, PATH_TO_ERIBERTA, PATH_TO_BASELINE_OUTPUT
import torch
import os

# https://huggingface.co/docs/transformers/main/en/tasks/sequence_classification


def delete_more_than_512_tokens(dataset):
    """
    Delete instances with more than 512 tokens
    """
    number_of_instances = 0
    for instance in dataset['input_ids']:
        if len(instance) > 512:
            dataset['input_ids'].remove(instance)
            number_of_instances += 1
    return dataset, number_of_instances


## 0. SET UP ##
# Set up hyperparameters
LR = 2e-5
EPOCHS = 1
BATCH_SIZE = 16
TYPE = 'wo_context' # set to 'with_context' or 'wo_context'

# Set up Weights & Biases
os.environ["WANDB_PROJECT"]="tfg-baseline" # set the wandb project where this run will be logged
os.environ["WANDB_LOG_MODEL"] = "checkpoint" # save your trained model checkpoint to wandb

# Set up device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")


## 1. LOAD DATASET ##
train_dataset = load_dataset_as_dict(PATH_TO_MEDMCQA_TRAIN)
dev_dataset = load_dataset_as_dict(PATH_TO_MEDMCQA_DEV)


## 2. TOKENIZATION ##

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(PATH_TO_ERIBERTA, use_fast=True)

# Tokenize and encode
print("Tokenizing and encoding datasets...")
encoded_train = tokenizer(train_dataset['inputs'], padding=True) # TODO: hace falta truncar?? Qué estrategia usamos??
print("\tTrain dataset tokenized and encoded. Number of instances: ", len(encoded_train['input_ids']))
encoded_dev = tokenizer(dev_dataset['inputs'], padding=True) # TODO: hace falta truncar?? Qué estrategia usamos??
print("\tDev dataset tokenized and encoded. Number of instances: ", len(encoded_dev['input_ids']))

# Get rid of instances with more than 512 tokens (TODO: probar otras estraegias)
encoded_train, number_of_deleted_instances = delete_more_than_512_tokens(encoded_train)
print("\tDeleted instances with more than 512 tokens from train dataset. Number of instances: ", len(encoded_train['input_ids']), "Deleted: ", number_of_deleted_instances)
encoded_dev, number_of_deleted_instances = delete_more_than_512_tokens(encoded_dev)
print("\tDeleted instances with more than 512 tokens from train dataset. Number of instances: ", len(encoded_dev['input_ids']), "Deleted: ", number_of_deleted_instances)


## 3. FINE-TUNE MODEL ##
# Load a torch.utils.data.Dataset object (required in Trainer)
encoded_train_dataset = MedMCQA_Dataset(encoded_train, train_dataset['labels'])
encoded_dev_dataset = MedMCQA_Dataset(encoded_dev, dev_dataset['labels'])

# Load model. Transformer with default classification head
model = AutoModelForSequenceClassification.from_pretrained(PATH_TO_ERIBERTA).to(device) 

# SET TRAINING HYPERPARAMETERS #
training_args = TrainingArguments(
    output_dir=PATH_TO_BASELINE_OUTPUT,         # output directory
    evaluation_strategy='steps',                # when to evaluate
    num_train_epochs=EPOCHS,                    # total number of training epochs
    per_device_train_batch_size=BATCH_SIZE,     # batch size per device during training
    per_device_eval_batch_size=BATCH_SIZE,      # batch size for evaluation
    warmup_steps=100,                           # number of warmup steps for learning rate scheduler
    #weight_decay=
    logging_dir='./logs',                       # directory for storing logs
    logging_steps=10,                           # when to print log (and to evaluate if evaluation_strategy = 'steps')
    load_best_model_at_end=True,                # load or not best model at the end
    #use_mps_device=True                        # para que use la GPU del MAC
    report_to='wandb'                           # report to Weights & Biases
)

# SET TRAINER: MODEL TO TRAIN, TRAINING ARGUMENTS, DATSET SPLITS...
trainer = Trainer(
    model=model,
    args=training_args,                   # training arguments, defined above
    train_dataset=encoded_train_dataset,  # training dataset. Format: torch.utils.data.Dataset
    eval_dataset=encoded_dev_dataset      # evaluation dataset. Format: torch.utils.data.Dataset
)

# FINE-TUNE MODEL ##
trainer.train()
