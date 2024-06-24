import matplotlib.pyplot as plt
from loadDataset import load_dataset_as_dict_with_context, load_dataset_as_dict_wo_context
from variables import PATH_TO_MEDMCQA_TRAIN, PATH_TO_MEDMCQA_DEV, PATH_TO_ERIBERTA
from transformers import AutoTokenizer
from tqdm import tqdm

# Load the data
# 0. SET UP PARAMETERS
TYPE = 'wo_context' # set to 'with_context' or 'wo_context'
load_dataset_as_dict = load_dataset_as_dict_with_context if TYPE == 'with_context' else load_dataset_as_dict_wo_context


# 1. LOAD DATASET
# Train and dev datasets
train_dataset = load_dataset_as_dict(PATH_TO_MEDMCQA_TRAIN)
dev_dataset = load_dataset_as_dict(PATH_TO_MEDMCQA_DEV)

# 2. TOKENIZATION
tokenizer = AutoTokenizer.from_pretrained(PATH_TO_ERIBERTA, use_fast=True) # TODO. Usar el modelo que debería usar, no cualquier cosa.
tokenized_and_encoded_train_dataset = tokenizer(train_dataset['inputs'])

tokenized_train_dataset = []
for input in tqdm(train_dataset['inputs'], desc="Tokenizing train dataset"):
    tokenized_train_dataset.append(tokenizer.tokenize(input))

tokenized_and_encoded_dev_dataset = tokenizer(dev_dataset['inputs'])

tokenized_instances_length_train = [len(instance) for instance in tokenized_and_encoded_train_dataset['input_ids']]
tokenized_instances_length_dev = [len(instance) for instance in tokenized_and_encoded_dev_dataset['input_ids']]

plt.boxplot(
    x=[tokenized_instances_length_train, tokenized_instances_length_dev], 
    labels=['train', 'dev'],
    vert=False,
    sym='+'
)
plt.savefig('boxplot.png')
