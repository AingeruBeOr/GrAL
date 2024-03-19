import os
import json
import sys
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import wandb
import evaluate
import numpy as np
from variables import PATH_TO_MEDMCQA_TRAIN, PATH_TO_MEDMCQA_DEV, PATH_TO_ERIBERTA, PATH_TO_BASELINE_OUTPUT, PATH_TO_MEDMCQA_LOADER

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
#print(dataset['train'][0]) # print first instance tokenized and encoded
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
    eval_steps=training_arguments['eval_steps'],                            # when to evaluate (and to print log if logging_strategy = 'steps')
    
    logging_dir=training_arguments['logging_dir'],                                   # directory for storing logs
    logging_steps=training_arguments['logging_steps'],                               # when to print log (and to evaluate if evaluation_strategy = 'steps')
    report_to=training_arguments['report_to'],                                       # report to Weights & Biases
    
    save_strategy=training_arguments['save_strategy'],                               # when to save each model checkpoint. It must be the same as 'evaluation_strategy' (porque guarda el modelo después de evaluarlo)
    save_steps=training_arguments['save_steps'],                                     # when to save each model checkpoint
    save_total_limit=training_arguments['save_total_limit'],                         # number of model checkpoints to keep
    load_best_model_at_end=training_arguments['load_best_model_at_end']              # load or not best model at the end
    
    #weight_decay=
    #use_mps_device=True                                                             # para que use la GPU del MAC
)

# SET TRAINER: MODEL TO TRAIN, TRAINING ARGUMENTS, DATSET SPLITS...
trainer = Trainer(
    model=model,
    args=training_args,              # training arguments, defined above
    train_dataset=dataset['train'],  # training dataset. Format: torch.utils.data.Dataset
    eval_dataset=dataset['dev'],     # evaluation dataset. Format: torch.utils.data.Dataset
    compute_metrics=compute_metrics  # evaluation function
)

# FINE-TUNE MODEL #d#
trainer.train()