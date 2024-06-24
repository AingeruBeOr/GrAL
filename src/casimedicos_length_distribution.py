from datasets import load_dataset
from transformers import AutoTokenizer
import pandas as pd
import plotly.express as px

dataloader = '../data/casiMedicos/casimedicos_data_loader.py'
data_files = {
    'train': 'JSONL/en.train_casimedicos.jsonl',
    'dev': 'JSONL/en.dev_casimedicos.jsonl',
    'test': 'JSONL/en.test_casimedicos.jsonl'
}
model_path = '../models/eriberta_libre'

# Load dataset
dataset = load_dataset(dataloader, data_files=data_files)
print(dataset['train'])

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Tokenize dataset
def generate_dataset_instances(instance):
    return {
        'instance': f"{instance['question']}</s>{instance['opa']}</s>{instance['opb']}</s>{instance['opc']}</s>{instance['opd']}</s>{instance['ope']}",
    }

get_tokenized_length = lambda x: tokenizer(x, padding=False, truncation=False, return_token_type_ids=False, return_attention_mask=False, return_length=True).length

def tokenize_function(examples):
    # Get the tokenized length for each instance
    return {
        'lengths': get_tokenized_length(examples['instance']),
    }

# Tokenize the dataset (every split)
tokenized_dataset_train_lengths = dataset['train'].map(generate_dataset_instances)
tokenized_dataset_train_lengths = tokenized_dataset_train_lengths.map(tokenize_function, batched=True)

tokenized_dataset_dev_lengths = dataset['dev'].map(generate_dataset_instances)
tokenized_dataset_dev_lengths = tokenized_dataset_dev_lengths.map(tokenize_function, batched=True)

tokenized_dataset_test_lengths = dataset['test'].map(generate_dataset_instances)
tokenized_dataset_test_lengths = tokenized_dataset_test_lengths.map(tokenize_function, batched=True)

train_lengths_df = pd.DataFrame(data = {'length': tokenized_dataset_train_lengths['lengths']})
train_lengths_df['split'] = 'Train'
dev_lengths_df = pd.DataFrame(data = {'length': tokenized_dataset_dev_lengths['lengths']})
dev_lengths_df['split'] = '<i>Validation</i>'
test_lengths_df = pd.DataFrame(data = {'length': tokenized_dataset_test_lengths['lengths']})
test_lengths_df['split'] = 'Test'

# Combine the train, dev and test dataframes
combined_df = pd.concat([train_lengths_df, dev_lengths_df, test_lengths_df], ignore_index=True)
print(combined_df)

# Plot the combined dataframe
fig = px.box(
    combined_df,
    x='length',
    y='split',
    color='split',
    title='Entrenamendu, validation eta test datu-sorten instantzien token luzera',
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
fig.write_image('../imgs/CasiMedicos_length_distribution.en.splits.png')