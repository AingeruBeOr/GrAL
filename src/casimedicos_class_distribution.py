from tqdm import tqdm
import matplotlib.pyplot as plt
import json

'''
Checks the distribution of the classes in the CasiMedicos dataset:

Outputs:
    - Print the class distribution of the train and dev datasets
    - Plot a bar chart with the class distribution of the train and dev datasets
'''

PATH_TO_CASIMEDICOS_TRAIN = '../data/casiMedicos/JSONL/en.train_casimedicos.jsonl'
PATH_TO_CASIMEDICOS_DEV = '../data/casiMedicos/JSONL/en.dev_casimedicos.jsonl'

train_distribution = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0}
with open(PATH_TO_CASIMEDICOS_TRAIN, 'r') as f:
    train_data = f.readlines()
    for instance in train_data:
        instance = json.loads(instance)
        if instance['correct_option'] == 1:
            train_distribution['A'] += 1
        elif instance['correct_option'] == 2:
            train_distribution['B'] += 1
        elif instance['correct_option'] == 3:
            train_distribution['C'] += 1
        elif instance['correct_option'] == 4:
            train_distribution['D'] += 1
        elif instance['correct_option'] == 5:
            train_distribution['E'] += 1

dev_distribution = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0}
with open(PATH_TO_CASIMEDICOS_DEV, 'r') as f:
    dev_data = f.readlines()
    for instance in dev_data:
        instance = json.loads(instance)
        if instance['correct_option'] == 1:
            dev_distribution['A'] += 1
        elif instance['correct_option'] == 2:
            dev_distribution['B'] += 1
        elif instance['correct_option'] == 3:
            dev_distribution['C'] += 1
        elif instance['correct_option'] == 4:
            dev_distribution['D'] += 1
        elif instance['correct_option'] == 5:
            dev_distribution['E'] += 1

print(f"\n\nTrain dataset class distribution: \n\n{train_distribution}")
print(f"\n\nDev dataset class distribution: \n\n{dev_distribution}")

plt.figure(1)
plt.bar(train_distribution.keys(), train_distribution.values(), color='b')
plt.title('Train dataset-aren erantzun zuzenen banaketa')
plt.xlabel('Aukerak')
for i, v in enumerate(train_distribution.values()):
    plt.text(i, v, str(v), ha='center', va='bottom')
plt.savefig('../imgs/CasiMedicos_train_class_distribution.png')

plt.figure(2)
plt.bar(dev_distribution.keys(), dev_distribution.values(), color='r')
plt.title('Dev dataset-aren erantzun zuzenen banaketa')
plt.xlabel('Aukerak')
for i, v in enumerate(dev_distribution.values()):
    plt.text(i, v, str(v), ha='center', va='bottom')
plt.savefig('../imgs/CasiMedicos_dev_class_distribution.png')