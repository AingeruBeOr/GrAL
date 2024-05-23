from tqdm import tqdm
import matplotlib.pyplot as plt
import json

DATASETS = [
    {
        'path': '../data/casiMedicos-balanced/JSONL/en.train_casimedicos.jsonl',
        'correct_option_label': 'correct_option'
    },
    {
        'path': '../data/MedMCQA-balanced/train.json',
        'correct_option_label': 'cop'
    },
]

train_distribution = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0}
for dataset in DATASETS:
    with open(dataset['path'], 'r') as f:
        train_data = f.readlines()
        for instance in train_data:
            instance = json.loads(instance)
            if instance[dataset['correct_option_label']] == 1:
                train_distribution['A'] += 1
            elif instance[dataset['correct_option_label']] == 2:
                train_distribution['B'] += 1
            elif instance[dataset['correct_option_label']] == 3:
                train_distribution['C'] += 1
            elif instance[dataset['correct_option_label']] == 4:
                train_distribution['D'] += 1
            elif instance[dataset['correct_option_label']] == 5:
                train_distribution['E'] += 1

print(f"\n\nTrain dataset class distribution: \n\n{train_distribution}")

plt.figure(1)
plt.bar(train_distribution.keys(), train_distribution.values(), color='b')
plt.title('Train dataset-aren erantzun zuzenen banaketa')
plt.xlabel('Aukerak')
for i, v in enumerate(train_distribution.values()):
    plt.text(i, v, str(v), ha='center', va='bottom')
plt.savefig('../imgs/CasiMedicos_and_medmcqa_train_class_distribution.png')
