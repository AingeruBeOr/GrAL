import json
from dataset import CasiMedicosDataset
from torch.utils.data.dataloader import DataLoader
import numpy as np
import os

'''
Script to generate JSONL files for the CasiMedicos-balanced dataset
'''

SETS_PATH = [
    '../../data/casiMedicos/JSONL/en.train_casimedicos.jsonl',
    '../../data/casiMedicos/JSONL/en.test_casimedicos.jsonl',
    '../../data/casiMedicos/JSONL/en.dev_casimedicos.jsonl'
]

for set_path in SETS_PATH:
    instances = []
    with open(set_path, 'r') as file:
        instances = [json.loads(row) for row in file]

    new_file_path = f'../../data/casiMedicos-balanced/JSONL/{os.path.basename(set_path)}'

    with open(new_file_path, 'w') as file:

        for instance in instances:
            id = instance['id']
            question = instance['full_question']
            correct_option_id = instance['correct_option']
            correct_option = instance['options'][str(correct_option_id)]
            incorrect_options = [instance['options'][index] for index, option in instance['options'].items() if index != correct_option_id]

            np.random.shuffle(incorrect_options)
            file.write(json.dumps({
                'id': f'{id}-1',
                'question': question,
                'options': {
                    "1": correct_option,
                    "2": incorrect_options[0],
                    "3": incorrect_options[1],
                    "4": incorrect_options[2],
                    "5": incorrect_options[3]
                },
                'correct_option': 1
            }) + '\n')

            np.random.shuffle(incorrect_options)
            file.write(json.dumps({
                'id': f'{id}-2',
                'question': question,
                'options': {
                    "1": incorrect_options[0],
                    "2": correct_option,
                    "3": incorrect_options[1],
                    "4": incorrect_options[2],
                    "5": incorrect_options[3]
                },
                'correct_option': 2
            }) + '\n')

            np.random.shuffle(incorrect_options)
            file.write(json.dumps({
                'id': f'{id}-3',
                'question': question,
                'options': {
                    "1": incorrect_options[0],
                    "2": incorrect_options[1],
                    "3": correct_option,
                    "4": incorrect_options[2],
                    "5": incorrect_options[3]
                },
                'correct_option': 3
            }) + '\n')

            np.random.shuffle(incorrect_options)
            file.write(json.dumps({
                'id': f'{id}-4',
                'question': question,
                'options': {
                    "1": incorrect_options[0],
                    "2": incorrect_options[1],
                    "3": incorrect_options[2],
                    "4": correct_option,
                    "5": incorrect_options[3]
                },
                'correct_option': 4
            }) + '\n')

            np.random.shuffle(incorrect_options)
            file.write(json.dumps({
                'id': f'{id}-5',
                'question': question,
                'options': {
                    "1": incorrect_options[0],
                    "2": incorrect_options[1],
                    "3": incorrect_options[2],
                    "4": incorrect_options[3],
                    "5": correct_option
                },
                'correct_option': 5
            }) + '\n') 

    print(f"File {new_file_path} created.")
