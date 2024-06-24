from tqdm import tqdm
from loadDataset import load_dataset_as_dict_with_context, load_dataset_as_dict_wo_context
from variables import PATH_TO_MEDMCQA_TRAIN, PATH_TO_MEDMCQA_DEV, PATH_TO_ERIBERTA
from transformers import AutoTokenizer


# 0. SET UP PARAMETERS
TYPE = 'wo_context' # set to 'with_context' or 'wo_context'
load_dataset_as_dict = load_dataset_as_dict_with_context if TYPE == 'with_context' else load_dataset_as_dict_wo_context


# 1. LOAD DATASET
# Train dataset
train_dataset = load_dataset_as_dict(PATH_TO_MEDMCQA_TRAIN)
print(f"Número de instancias en train: {len(train_dataset['inputs'])}")
print(f"\nInstancia 0:")
print(f"\tInput: {train_dataset['inputs'][0]}")
print(f"\tLabel (correct option): {train_dataset['labels'][0]}")

# Dev dataset
dev_dataset = load_dataset_as_dict(PATH_TO_MEDMCQA_DEV)
print(f"\nNúmero de instancias en dev: {len(dev_dataset['inputs'])}")


# 2. TOKENIZATION
# Preparamos el tokenizador:
tokenizer = AutoTokenizer.from_pretrained(PATH_TO_ERIBERTA, use_fast=True) # TODO. Usar el modelo que debería usar, no cualquier cosa.

# Indicamos cuales son los tokens especiales para que no los parta (no hace falta en EriBERTa porque ya son sus tokens especiales)
"""special_tokens_dict = {
    'bos_token': '<s>',
    'sep_token': '</s>',
    'eos_token': '</s>'
    }
tokenizer.add_special_tokens(special_tokens_dict)"""
tokenized_and_encoded_train_dataset = tokenizer(train_dataset['inputs'])

tokenized_train_dataset = []
for input in tqdm(train_dataset['inputs'], desc="Tokenizing train dataset"):
    tokenized_train_dataset.append(tokenizer.tokenize(input))

tokenized_and_encoded_dev_dataset = tokenizer(dev_dataset['inputs'])


# Comprobación de una instancia de `train`:
# - Visualizar los tokens en formato text
# - Visualizar los tokens en formato id
print(f"Example tokenized instance (train):")
print(f"\tTokens: {tokenized_train_dataset[0]}")
print(f"\tNumber of tokens: {len(tokenized_train_dataset[0])}")
print(f"\tToken ids: {tokenized_and_encoded_train_dataset['input_ids'][0]}")
print(f"\tNumber of token ids: {len(tokenized_and_encoded_train_dataset['input_ids'][0])}")


# 3. ESTADÍSTICAS
# 3.1. Distribución de número de tokens por split (train y dev) en boxplot
import pandas as pd
import plotly.graph_objects as go

df_train = pd.DataFrame({'number_of_tokens': [len(instance) for instance in tokenized_and_encoded_train_dataset['input_ids']]})
df_dev = pd.DataFrame({'number_of_tokens': [len(instance) for instance in tokenized_and_encoded_dev_dataset['input_ids']]})

# Crea el primer boxplot
trace1 = go.Box(
    x=df_train['number_of_tokens'],
    name='Train',
    boxpoints='outliers'               # If 'outliers', show only the outliers points
)

# Crea el segundo boxplot
trace2 = go.Box(
    x=df_dev['number_of_tokens'],
    name='Validation',
    boxpoints='outliers'               # If 'outliers', show only the outliers points
)

# Combina los boxplots en una sola figura
data = [trace1, trace2]
title = "Distribution of the number of tokens in the input (<s>Context</s>...)" if TYPE == 'with_context' else "Distribution of the number of tokens in the input"
layout = go.Layout(title="Distribution of the number of tokens in the input (<s>Context</s>...)", xaxis_title="Number of tokens", yaxis_title="Split", xaxis=dict(type='log', autorange=True))

fig = go.Figure(data=data, layout=layout)
fig.show()


# 3.2. Comprobamos si hay alguna instancia con más de 512 tokens e imprimimos ejemplos:

def more_than_512_tokens(tokenized_dataset):
    number_of_more_than_512_tokens = 0
    for instance_tokenized in tokenized_dataset['input_ids']:
        if len(instance_tokenized) > 512:
            number_of_more_than_512_tokens += 1
    return number_of_more_than_512_tokens


# En el split train
more_than_512_tokens_train = more_than_512_tokens(tokenized_and_encoded_train_dataset)
print(f"Number of instances with more than 512 tokens (train): {more_than_512_tokens_train}/{len(tokenized_and_encoded_train_dataset['input_ids'])} ({more_than_512_tokens_train/len(tokenized_and_encoded_train_dataset['input_ids'])*100:.2f}%)")

number_of_examples = 10
for i, instance_tokenized in enumerate(tokenized_and_encoded_train_dataset['input_ids']):
        if len(instance_tokenized) > 512:
            instance_split = train_dataset['inputs'][i].split('</s>')
            print(f"Instance {i}:\n\t Context: {instance_split[0].replace('<s>', '')} \n\t Question: {instance_split[1]} \n\t Option A: {instance_split[2]} \n\t Option B: {instance_split[3]} \n\t Option C: {instance_split[4]} \n\t Option D: {instance_split[5]} \n\t Correct option: {train_dataset['labels'][i]}")
            number_of_examples -= 1
            if number_of_examples == 0:
                break

# En el split dev
more_than_512_tokens_dev = more_than_512_tokens(tokenized_and_encoded_dev_dataset)
print(f"Number of instances with more than 512 tokens (dev): {more_than_512_tokens_dev}/{len(tokenized_and_encoded_dev_dataset['input_ids'])} ({more_than_512_tokens_dev/len(tokenized_and_encoded_dev_dataset['input_ids'])*100:.2f}%)")


# 3.3. Boxplot de diferentes partes de los inputs:
encoded_train_context = []
encoded_train_question = []
encoded_train_optiona = []
encoded_train_optionb = []
encoded_train_optionc = []
encoded_train_optiond = []
encoded_train_options = []

for input in tqdm(train_dataset['inputs'], desc="Tokenizing different parts of the input"):
    if TYPE == 'with_context':
        context = input.split('</s>')[0]
        question = input.split('</s>')[1]
        optiona = input.split('</s>')[2]
        optionb = input.split('</s>')[3]
        optionc = input.split('</s>')[4]
        optiond = input.split('</s>')[5]
        options = optiona + '</s>' + optionb + '</s>' + optionc + '</s>' + optiond
    else:
        question = input.split('</s>')[0]
        optiona = input.split('</s>')[1]
        optionb = input.split('</s>')[2]
        optionc = input.split('</s>')[3]
        optiond = input.split('</s>')[4]
        options = optiona + '</s>' + optionb + '</s>' + optionc + '</s>' + optiond

    if TYPE == 'with_context': encoded_train_context.append(tokenizer(context))
    encoded_train_question.append(tokenizer(question))
    encoded_train_optiona.append(tokenizer(optiona))
    encoded_train_optionb.append(tokenizer(optionb))
    encoded_train_optionc.append(tokenizer(optionc))
    encoded_train_optiond.append(tokenizer(optiond))
    encoded_train_options.append(tokenizer(options))

print(len(encoded_train_question))


if TYPE == 'with_context': df_train_context = pd.DataFrame({'number_of_tokens': [len(instance['input_ids']) for instance in encoded_train_context]})
df_train_question = pd.DataFrame({'number_of_tokens': [len(instance['input_ids']) for instance in encoded_train_question]})
df_train_optiona = pd.DataFrame({'number_of_tokens': [len(instance['input_ids']) for instance in encoded_train_optiona]})
df_train_optionb = pd.DataFrame({'number_of_tokens': [len(instance['input_ids']) for instance in encoded_train_optionb]})
df_train_optionc = pd.DataFrame({'number_of_tokens': [len(instance['input_ids']) for instance in encoded_train_optionc]})
df_train_optiond = pd.DataFrame({'number_of_tokens': [len(instance['input_ids']) for instance in encoded_train_optiond]})
df_train_options = pd.DataFrame({'number_of_tokens': [len(instance['input_ids']) for instance in encoded_train_options]})

if TYPE == 'with_context':
    # Crea el primer boxplot
    trace1 = go.Box(
        x=df_train_context['number_of_tokens'],
        name='Train context'
    )

# Crea el segundo boxplot
trace2 = go.Box(
    x=df_train_question['number_of_tokens'],
    name='Train question'
)

# Crea el tercer boxplot
trace3 = go.Box(
    x=df_train_optiona['number_of_tokens'],
    name='Train option A'
)

# Crea el cuarto boxplot
trace4 = go.Box(
    x=df_train_optionb['number_of_tokens'],
    name='Train option B'
)

# Crea el quinto boxplot
trace5 = go.Box(
    x=df_train_optionc['number_of_tokens'],
    name='Train option C'
)

# Crea el sexto boxplot
trace6 = go.Box(
    x=df_train_optiond['number_of_tokens'],
    name='Train option D'
)

# Crea el séptimo boxplot
trace7 = go.Box(
    x=df_train_options['number_of_tokens'],
    name='Train options'
)

# Combina los boxplots en una sola figura
if TYPE == 'with_context': data = [trace1, trace2, trace3, trace4, trace5, trace6, trace7]
else: data = [trace2, trace3, trace4, trace5, trace6, trace7]
layout = go.Layout(
    title="Distribution of the number of tokens in the input", 
    xaxis_title="Number of tokens", 
    yaxis_title="Split", 
    xaxis=dict(type='log', autorange=True)
    )

fig = go.Figure(data=data, layout=layout)
fig.show()