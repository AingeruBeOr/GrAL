import logging
import os
import shutil
import sys

import wandb.plot
from ModelDataTrainingArguments import ModelDataTrainingArguments
from eval_function import compute_metrics
import datasets
from datasets import load_dataset
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, set_seed, HfArgumentParser
import wandb
import numpy as np
import datetime

'''
This script is used to train a model using baseline architecture for the Casimedicos(en) dataset.

https://huggingface.co/docs/transformers/main/en/tasks/sequence_classification
'''

logger = logging.getLogger(__name__)

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


# Set up logger
def setup_logger(training_args):
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    #logger.info(f"Training/evaluation parameters {training_args}")


def main(training_arguments_path: str):
    parser = HfArgumentParser((ModelDataTrainingArguments, TrainingArguments))
    model_data_args: ModelDataTrainingArguments
    training_args: TrainingArguments
    model_data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(training_arguments_path))

    # Create output directory with current timestamp
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d-%H-%M")
    training_args.output_dir = training_args.output_dir + f"{formatted_datetime}"

    # Create directory and save (Copy) the training arguments json in this directory
    os.makedirs(training_args.output_dir, exist_ok=False)
    shutil.copyfile(training_arguments_path, training_args.output_dir + '/training_arguments.json')

    # Set up Weights & Biases
    wandb.init(
        project="tfg-baseline-casimedicos",
        name=formatted_datetime,
    )
    os.environ["WANDB_LOG_MODEL"] = "false"  # do not upload the model to wandb

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
    model = AutoModelForSequenceClassification.from_pretrained(
        model_data_args.model_name_or_path,
        num_labels=num_labels,
        id2label=id_to_label,
        label2id=label_to_id,
    )

    # SET TRAINER: MODEL TO TRAIN, TRAINING ARGUMENTS, DATSET SPLITS...
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,  # training arguments, defined above
        train_dataset=dataset['train'],  # training dataset. Format: torch.utils.data.Dataset
        eval_dataset=dataset['dev'],  # evaluation dataset. Format: torch.utils.data.Dataset
        compute_metrics=compute_metrics  # evaluation function
    )

    # FINE-TUNE MODEL #d#
    train_result = trainer.train()
    train_metrics = train_result.metrics
    train_metrics["train_samples"] = len(dataset['train'])
    trainer.save_model()  # Saves the tokenizer too for easy upload
    trainer.log_metrics(split="train", metrics=train_metrics) # Combined 'False' to avoid creating 'all_results.json'
    trainer.save_metrics(split="train", metrics=train_metrics, combined=False) # Creates: <output_dir>/train_results.json # Combined 'False' to avoid creating 'all_results.json'
    trainer.save_state()

    # EVALUATE MODEL (INFERENCE) #
    logger.info("*** Evaluate ***")
    inference_splits = {
        "inference_eval": dataset['dev'],
        "inference_test": dataset['test']
    }
    for split_name, instances in inference_splits.items():
        results = trainer.predict(test_dataset=instances) # .predict() is used insted of .evaluate() to get predictions for the confusion matrix
        metrics = results.metrics
        predictions = results.predictions
        predictions = np.argmax(predictions, axis=1)
        metrics["samples"] = len(instances)
        metrics.pop("test_confusion_matrix") # remove the confusion matrix from the metrics (tries to serialize it and fails)
        confusion_matrix = wandb.plot.confusion_matrix(probs=None, y_true=results.label_ids, preds=predictions, class_names=label_list, title=f'{split_name} confusion matrix')
        wandb.log({f'{split_name}_confusion_matrix': confusion_matrix})
        trainer.log_metrics(split=split_name, metrics=metrics)  # Log # Combined 'False' to avoid creating 'all_results.json'
        trainer.save_metrics(split=split_name, metrics=metrics, combined=False) # Creates: <output_dir>/<split_name>_results.json # Combined 'False' to avoid creating 'all_results.json'


if __name__ == '__main__':
    ## 0. SET UP ##
    # Load hyperparameters from json file
    assert len(sys.argv) == 2, "Usage: python train.py <path_to_json_file>"
    main(sys.argv[1])
