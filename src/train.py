import os
from variables import PATH_TO_MEDMCQA_TRAIN, PATH_TO_MEDMCQA_DEV, PATH_TO_ERIBERTA, PATH_TO_BASELINE_OUTPUT, PATH_TO_MEDMCQA_LOADER
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import torch

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
LR = 5e-5               # This is the maximum learning rate value. In the warm up it will be adjusted. 
EPOCHS = 5              # Number of epochs
BATCH_SIZE_TRAIN = 12   # Batch size for training. Lo máximo que entre en la GPU
BATCH_SIZE_DEV = 16     # Batch size for evaluation. Lo máximo que entre en la GPU. Suele entrar más porque no hay backpropagation

# Set up Weights & Biases
os.environ["WANDB_PROJECT"] = "tfg-baseline" # set the wandb project where this run will be logged
os.environ["WANDB_LOG_MODEL"] = "checkpoint" # save your trained model checkpoint to wandb

# Set up device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")


## 1. LOAD DATASET ##
dataset = load_dataset(path = PATH_TO_MEDMCQA_LOADER, 
                       data_files={"train": PATH_TO_MEDMCQA_TRAIN, 
                                   "dev": PATH_TO_MEDMCQA_DEV}
                       )
print(f"Dataset structure: \n\n{dataset}")
print(f"\n\nDataset train first instance example: \n\n{dataset['train'][0]}")

## 2. TOKENIZATION ##
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(PATH_TO_ERIBERTA, use_fast=True)

# Preprecess function
def preprocess_function(example):
    my_input = f'{example["question"]}</s>{example["opa"]}</s>{example["opb"]}</s>{example["opc"]}</s>{example["opd"]}'
    return tokenizer(my_input, padding='max_length', max_length=512)

# Tokenize and encode
print("\n\nTokenizing and encoding datasets...")
dataset = dataset.map(preprocess_function) # TODO poner batched True
print(dataset['train'][0])
print("\tTrain dataset tokenized and encoded. Number of instances: ", len(dataset['train']))
print("\tDev dataset tokenized and encoded. Number of instances: ", len(dataset['dev']), "\n\n")

# Get rid of instances with more than 512 tokens (TODO: probar otras estraegias)
"""encoded_train, number_of_deleted_instances = delete_more_than_512_tokens(encoded_train)
print("\tDeleted instances with more than 512 tokens from train dataset. Number of instances: ", len(encoded_train['input_ids']), "Deleted: ", number_of_deleted_instances)
encoded_dev, number_of_deleted_instances = delete_more_than_512_tokens(encoded_dev)
print("\tDeleted instances with more than 512 tokens from train dataset. Number of instances: ", len(encoded_dev['input_ids']), "Deleted: ", number_of_deleted_instances)
"""

## 3. FINE-TUNE MODEL ##
# Load model. Transformer with default classification head
model = AutoModelForSequenceClassification.from_pretrained(PATH_TO_ERIBERTA, num_labels=4) 

# SET TRAINING HYPERPARAMETERS #
training_args = TrainingArguments(
    output_dir=PATH_TO_BASELINE_OUTPUT,             # output directory. Carpeta donde se generarán todos los ficheros de salida y checkpoints. (No es necesario que la carpeta exista, el script crea todas las carpetas necesarias)
    evaluation_strategy='steps',                    # when to evaluate. Options: 'no', 'steps', 'epoch'
    eval_steps=100,                                 # when to evaluate (and to print log if logging_strategy = 'steps')
    num_train_epochs=EPOCHS,                        # total number of training epochs
    per_device_train_batch_size=BATCH_SIZE_TRAIN,   # batch size per device during training
    per_device_eval_batch_size=BATCH_SIZE_DEV,      # batch size for evaluation
    gradient_accumulation_steps=5,                  # gradient accumulation to avoid OOM errors
    warmup_steps=100,                               # number of warmup steps for learning rate scheduler
    #weight_decay=
    logging_dir='./logs',                           # directory for storing logs
    logging_steps=10,                               # when to print log (and to evaluate if evaluation_strategy = 'steps')
    report_to='wandb',                              # report to Weights & Biases
    save_strategy='steps',                          # when to save each model checkpoint. It must be the same as 'evaluation_strategy' (porque guarda el modelo después de evaluarlo)
    save_steps=3000,                                # when to save each model checkpoint
    save_total_limit=3,                             # number of model checkpoints to keep
    load_best_model_at_end=True                     # load or not best model at the end
    #use_mps_device=True                            # para que use la GPU del MAC
)

# SET TRAINER: MODEL TO TRAIN, TRAINING ARGUMENTS, DATSET SPLITS...
trainer = Trainer(
    model=model,
    args=training_args,              # training arguments, defined above
    train_dataset=dataset['train'],  # training dataset. Format: torch.utils.data.Dataset
    eval_dataset=dataset['dev']      # evaluation dataset. Format: torch.utils.data.Dataset
)

# FINE-TUNE MODEL #d#
trainer.train()
