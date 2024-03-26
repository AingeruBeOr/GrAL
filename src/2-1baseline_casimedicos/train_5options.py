import logging
import os
import shutil
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import evaluate
import numpy as np
import transformers
import wandb
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, set_seed, HfArgumentParser

# https://huggingface.co/docs/transformers/main/en/tasks/sequence_classification
logger = logging.getLogger(__name__)

# Set up Weights & Biases
os.environ["WANDB_PROJECT"] = "tfg-baseline-casimedicos"  # set the wandb project where this run will be logged
os.environ["WANDB_LOG_MODEL"] = "false"  # do not upload the model to wandb


@dataclass
class ModelDataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    data_loader: str = field(
        metadata={"help": "Path to the data loader script."}
    )
    train_file: str = field(
        metadata={"help": "File containing the training data."}
    )
    validation_file: str = field(
        metadata={"help": "File containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "File containing the test data."})

    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )


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
    logger.info(f"Training/evaluation parameters {training_args}")


def main(training_arguments_path: str):
    parser = HfArgumentParser((ModelDataTrainingArguments, TrainingArguments))
    model_data_args: ModelDataTrainingArguments
    training_args: TrainingArguments
    model_data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(training_arguments_path))

    # Set up logger
    setup_logger(training_args)

    # Set seed
    set_seed(training_args.seed)

    ## 1. LOAD DATASET ##
    dataset = load_dataset(
        path=model_data_args.data_loader,
        data_files={
            "train": model_data_args.train_file,
            "dev": model_data_args.validation_file
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

    # Set evaluetaion function to be used during training (accuracy, f1, ...)
    accuracy = evaluate.load('accuracy')  # Load the accuracy function
    f1 = evaluate.load('f1')  # Load the f-score function
    precision = evaluate.load('precision')  # Load the precision function
    recall = evaluate.load('recall')  # Load the recall function
    cnf_matrix = evaluate.load('BucketHeadP65/confusion_matrix')  # Load the confusion matrix function

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

        #  Every element in the return dict, must be serializable so we log confusion_matrix (which is a plot object from wanbd library) independently to wandb
        confusion_matrix = wandb.plot.confusion_matrix(probs=None, y_true=labels, preds=predictions, class_names=label_list)
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
            'confusion_matrix': confusion_matrix_serializable['confusion_matrix'].tolist()  # convert to list to be serializable
        }

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
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)
    trainer.save_state()

    # Save (Copy) the training arguments json in the output directory
    shutil.copyfile(training_arguments_path, training_args.output_dir + '/training_arguments.json')

    logger.info("*** Evaluate ***")
    metrics = trainer.evaluate(eval_dataset=dataset['dev'])
    metrics["eval_samples"] = len(dataset['dev'])
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)




if __name__ == '__main__':
    ## 0. SET UP ##
    # Load hyperparameters from json file
    assert len(sys.argv) == 2, "Usage: python train.py <path_to_json_file>"
    main(sys.argv[1])
