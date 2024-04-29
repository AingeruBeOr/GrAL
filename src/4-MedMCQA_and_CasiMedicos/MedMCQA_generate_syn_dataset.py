import json
from dataset import MedMCQA4_json
from torch.utils.data.dataloader import DataLoader
import numpy as np
import os
from tqdm import tqdm

SETS_PATH = [
    '../../data/MedMCQA/train.json'
]

for set_path in SETS_PATH:
    set = MedMCQA4_json(set_path)

    new_file_path = f'../../data/MedMCQA-balanced/{os.path.basename(set_path)}'

    with open(new_file_path, 'w') as file:
        dataloader = DataLoader(set, batch_size=1, shuffle=False)   

        for instance in tqdm(dataloader, "Generating new dataset"):
            question = instance[0][0]
            correct_option_id = instance[2].item()
            correct_option = instance[1][correct_option_id][0]
            incorrect_options = [instance[1][index][0] for index, option in enumerate(instance[1]) if index != correct_option_id]

            np.random.shuffle(incorrect_options)
            file.write(json.dumps({
                'question': question,
                'opa': correct_option,
                'opb': incorrect_options[0],
                'opc': incorrect_options[1],
                'opd': incorrect_options[2],
                'cop': 1
            }) + '\n')

            np.random.shuffle(incorrect_options)
            file.write(json.dumps({
                'question': question,
                'opa': incorrect_options[0],
                'opb': correct_option,
                'opc': incorrect_options[1],
                'opd': incorrect_options[2],
                'cop': 2
            }) + '\n')

            np.random.shuffle(incorrect_options)
            file.write(json.dumps({
                'question': question,
                'opa': incorrect_options[0],
                'opb': incorrect_options[1],
                'opc': correct_option,
                'opd': incorrect_options[2],
                'cop': 3
            }) + '\n')

            np.random.shuffle(incorrect_options)
            file.write(json.dumps({
                'question': question,
                'opa': incorrect_options[0],
                'opb': incorrect_options[1],
                'opc': incorrect_options[2],
                'opd': correct_option,
                'cop': 4
            }) + '\n')

print(f"File {new_file_path} created.")
