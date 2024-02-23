from torch.utils.data import Dataset
import json
import torch

class MedMCQA_Dataset(Dataset):
    def __init__(self, encodings: list, labels: list):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, index):
        item = {key: torch.tensor(val[index]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[index])
        return item
    

def load_dataset_as_dict_with_context(path):
    with open(path, 'r') as file:
        dataset = [json.loads(row) for row in file]

    # Formato de dataset_dict:
    # {'inputs': ['<s>exp</s>question</s>opa</s>opb</s>opc</s>opd', ...], 'labels': [1, 2, 3, 4, ...]}
    dataset_dict = {}
    # No hace falta poner los tokens de inicio y fin de secuencia porque el tokenizador ya los añade automáticamente
    dataset_dict['inputs'] = [f'{instance["exp"]}</s>{instance["question"]}</s>{instance["opa"]}</s>{instance["opb"]}</s>{instance["opc"]}</s>{instance["opd"]}' for instance in dataset]
    dataset_dict['labels'] = [instance["cop"] for instance in dataset]

    return dataset_dict


def load_dataset_as_dict_wo_context(path):
    with open(path, 'r') as file:
        dataset = [json.loads(row) for row in file]

    # Formato de dataset_dict:
    # {'inputs': ['<s>question</s>opa</s>opb</s>opc</s>opd', ...], 'labels': [1, 2, 3, 4, ...]}
    dataset_dict = {}
    # No hace falta poner los tokens de inicio y fin de secuencia porque el tokenizador ya los añade automáticamente
    dataset_dict['inputs'] = [f'{instance["question"]}</s>{instance["opa"]}</s>{instance["opb"]}</s>{instance["opc"]}</s>{instance["opd"]}' for instance in dataset]
    dataset_dict['labels'] = [instance["cop"] for instance in dataset]

    return dataset_dict
