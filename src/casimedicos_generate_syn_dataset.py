import json
import numpy as np
import os

'''
Script to generate JSONL files for the CasiMedicos-balanced dataset
'''

LANG = 'es'
SETS_PATH = [
    f'../data/casiMedicos/JSONL/{LANG}.train_casimedicos.jsonl',
    f'../data/casiMedicos/JSONL/{LANG}.test_casimedicos.jsonl',
    f'../data/casiMedicos/JSONL/{LANG}.dev_casimedicos.jsonl'
]
np.random.seed(42) # Seed for reproducibility

for set_path in SETS_PATH:
    instances = []
    with open(set_path, 'r') as file:
        instances = [json.loads(row) for row in file]

    new_file_path = f'../data/casiMedicos-balanced/JSONL/{os.path.basename(set_path)}'

    with open(new_file_path, 'w') as file:

        for instance in instances:
            id = instance['id']
            question = instance['full_question']
            correct_option_id = instance['correct_option']
            correct_option = instance['options'][str(correct_option_id)]
            incorrect_options = [instance['options'][index] for index, option in instance['options'].items() if int(index) != correct_option_id]

            np.random.shuffle(incorrect_options)
            file.write(json.dumps({
                'id': f'{id}-1',
                'question': question,
                'options': {
                    "1": correct_option,
                    "2": incorrect_options[0] if isinstance(incorrect_options[0], str) else "",
                    "3": incorrect_options[1] if isinstance(incorrect_options[1], str) else "",
                    "4": incorrect_options[2] if isinstance(incorrect_options[2], str) else "",
                    "5": incorrect_options[3] if isinstance(incorrect_options[3], str) else ""
                },
                'correct_option': 1
            }) + '\n')

            np.random.shuffle(incorrect_options)
            file.write(json.dumps({
                'id': f'{id}-2',
                'question': question,
                'options': {
                    "1": incorrect_options[0] if isinstance(incorrect_options[0], str) else "",
                    "2": correct_option,
                    "3": incorrect_options[1] if isinstance(incorrect_options[1], str) else "",
                    "4": incorrect_options[2] if isinstance(incorrect_options[2], str) else "",
                    "5": incorrect_options[3] if isinstance(incorrect_options[3], str) else ""
                },
                'correct_option': 2
            }) + '\n')

            np.random.shuffle(incorrect_options)
            file.write(json.dumps({
                'id': f'{id}-3',
                'question': question,
                'options': {
                    "1": incorrect_options[0] if isinstance(incorrect_options[0], str) else "",
                    "2": incorrect_options[1] if isinstance(incorrect_options[1], str) else "",
                    "3": correct_option,
                    "4": incorrect_options[2] if isinstance(incorrect_options[2], str) else "",
                    "5": incorrect_options[3] if isinstance(incorrect_options[3], str) else ""
                },
                'correct_option': 3
            }) + '\n')

            np.random.shuffle(incorrect_options)
            file.write(json.dumps({
                'id': f'{id}-4',
                'question': question,
                'options': {
                    "1": incorrect_options[0] if isinstance(incorrect_options[0], str) else "",
                    "2": incorrect_options[1] if isinstance(incorrect_options[1], str) else "",
                    "3": incorrect_options[2] if isinstance(incorrect_options[2], str) else "",
                    "4": correct_option,
                    "5": incorrect_options[3] if isinstance(incorrect_options[3], str) else ""
                },
                'correct_option': 4
            }) + '\n')

            np.random.shuffle(incorrect_options)
            file.write(json.dumps({
                'id': f'{id}-5',
                'question': question,
                'options': {
                    "1": incorrect_options[0] if isinstance(incorrect_options[0], str) else "",
                    "2": incorrect_options[1] if isinstance(incorrect_options[1], str) else "",
                    "3": incorrect_options[2] if isinstance(incorrect_options[2], str) else "",
                    "4": incorrect_options[3] if isinstance(incorrect_options[3], str) else "",
                    "5": correct_option
                },
                'correct_option': 5
            }) + '\n') 

    print(f"File {new_file_path} created.")
