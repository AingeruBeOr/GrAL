from datasets import load_dataset
from variables import PATH_TO_MEDMCQA_LOADER, PATH_TO_MEDMCQA_TRAIN, PATH_TO_MEDMCQA_DEV
from tqdm import tqdm
import matplotlib.pyplot as plt

dataset = load_dataset(path=PATH_TO_MEDMCQA_LOADER,
                       data_files={"train": PATH_TO_MEDMCQA_TRAIN,
                                   "dev": PATH_TO_MEDMCQA_DEV}
                       )

print(f"Dataset structure: \n\n{dataset}")
train_distribution = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
for instance in tqdm(dataset['train'], desc="Counting class distribution in train"):
    if instance['label'] == 0:
        train_distribution['A'] += 1
    elif instance['label'] == 1:
        train_distribution['B'] += 1
    elif instance['label'] == 2:
        train_distribution['C'] += 1
    elif instance['label'] == 3:
        train_distribution['D'] += 1

dev_distribution = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
for instance in tqdm(dataset['dev'], desc="Counting class distribution in dev"):
    if instance['label'] == 0:
        dev_distribution['A'] += 1
    elif instance['label'] == 1:
        dev_distribution['B'] += 1
    elif instance['label'] == 2:
        dev_distribution['C'] += 1
    elif instance['label'] == 3:
        dev_distribution['D'] += 1

print(f"\n\nTrain dataset class distribution: \n\n{train_distribution}")
print(f"\n\nDev dataset class distribution: \n\n{dev_distribution}")

plt.figure(1)
plt.bar(train_distribution.keys(), train_distribution.values(), color='b')
plt.title('Train dataset-aren erantzun zuzenen banaketa')
plt.xlabel('Aukerak')
for i, v in enumerate(train_distribution.values()):
    plt.text(i, v, str(v), ha='center', va='bottom')
plt.savefig('../../imgs/MedMCQA_train_class_distribution-eu.png')

plt.figure(2)
plt.bar(dev_distribution.keys(), dev_distribution.values(), color='r')
plt.title('Dev dataset-aren erantzun zuzenen banaketa')
plt.xlabel('Aukerak')
for i, v in enumerate(dev_distribution.values()):
    plt.text(i, v, str(v), ha='center', va='bottom')
plt.savefig('../../imgs/MedMCQA_dev_class_distribution-eu.png')