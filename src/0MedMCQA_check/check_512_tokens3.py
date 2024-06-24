from loadDataset import load_dataset_as_dict_with_context, load_dataset_as_dict_wo_context
from variables import PATH_TO_MEDMCQA_TRAIN, PATH_TO_MEDMCQA_DEV, PATH_TO_ERIBERTA
from transformers import AutoTokenizer
from tqdm import tqdm
import pandas as pd
import plotly.express as px
import plotly.io as pio


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

# 3. PLOT
train_lengths = [len(instance) for instance in tokenized_and_encoded_train_dataset['input_ids']]
train_lengths_df = pd.DataFrame(data = {'length': train_lengths})
train_lengths_df['split'] = 'Train'
dev_lengths = [len(instance) for instance in tokenized_and_encoded_dev_dataset['input_ids']]
dev_lengths_df = pd.DataFrame(data = {'length': dev_lengths})
dev_lengths_df['split'] = '<i>Validation</i>'

# Combine the train and dev dataframes
combined_df = pd.concat([train_lengths_df, dev_lengths_df], ignore_index=True)
print(combined_df)

# Plot the combined dataframe
fig = px.box(
    combined_df,
    x='length',
    y='split',
    color='split',
    title='Entrenamendu eta <i>validation</i> datu-sorten instantzien token luzera',
    labels={'luzera': 'Tokenized length', 'multzoa': 'instantzia multzoa'}
)

# Add a red line for the maximum length (512)
fig.add_shape(
    type="line",
    x0=512,
    y0=0,
    x1=512,
    y1=1,
    xref='x',
    yref='paper',
    line=dict(color="Red", width=2)
)

# Change axes titles and Remove legend
fig.update_layout(
    xaxis_title='Token luzera', 
    yaxis_title='Datu-sorta multzoa', 
    showlegend=False
)

# Change X axis to log scale
fig.update_xaxes(type="log")

# Save the plot
fig.write_image('../../imgs/MedMCQA_length_distribution.png')