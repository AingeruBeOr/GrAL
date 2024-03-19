from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

SPLITS = ['train', 'dev', 'test']
BASE_PATH = Path('/home/shared/esperimentuak/AingeruTFG/TFG')


def check_lengths_casimedicos(model_path, data_files, dataloader):
    # Load dataset
    dataset = load_dataset(dataloader, data_files=data_files)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Tokenize dataset
    def generate_dataset_instances(instance):
        return {
            'options': f"{instance['opa']}</s>{instance['opb']}</s>{instance['opc']}</s>{instance['opd']}</s>{instance['ope']}",
            'question+options': f"{instance['question']}</s>{instance['opa']}</s>{instance['opb']}</s>{instance['opc']}</s>{instance['opd']}</s>{instance['ope']}",
            'full_instance': f"{instance['question']}</s>{instance['opa']}</s>{instance['opb']}</s>{instance['opc']}</s>{instance['opd']}</s>{instance['ope']}</s>{instance['answer_justification']}"
        }

    get_tokenized_length = lambda x: tokenizer(x, padding=False, truncation=False, return_token_type_ids=False, return_attention_mask=False, return_length=True).length

    def tokenize_function(examples):
        # Get the multiple input_ids for each instance combinations: question, options, question+options, full_instance, ...
        return {
            'question': get_tokenized_length(examples['question']),
            'options': get_tokenized_length(examples['options']),
            'question+options': get_tokenized_length(examples['question+options']),
            'full_instance': get_tokenized_length(examples['full_instance']),
            'answer_justification': get_tokenized_length(examples['answer_justification']),
            'opa': get_tokenized_length(examples['opa']),
            'opb': get_tokenized_length(examples['opb']),
            'opc': get_tokenized_length(examples['opc']),
            'opd': get_tokenized_length(examples['opd']),
            'ope': get_tokenized_length(examples['ope']),
        }

    tokenized_dataset_lengths = dataset.map(generate_dataset_instances)
    tokenized_dataset_lengths = tokenized_dataset_lengths.map(tokenize_function, batched=True)

    # Convert to numpy
    # TODO


if __name__ == '__main__':
    check_lengths_casimedicos(
        model_path=str(BASE_PATH / 'models/eriberta_libre'),
        data_files=[
            str(BASE_PATH / f"data/casiMedicos/JSONL/en.{_split}_casimedicos.jsonl")
            for _split in SPLITS
        ],
        dataloader=str(BASE_PATH / 'data/casiMedicos/casimedicos_data_loader.py')
    )
