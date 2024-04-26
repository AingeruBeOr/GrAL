from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.io as pio
from datasets import load_dataset
from transformers import AutoTokenizer

# Default theme
pio.templates.default = "plotly_white"

SPLITS = ['train', 'dev', 'test']
BASE_PATH = Path('./')


def check_lengths_casimedicos(model_path, data_files, dataloader, split, language):
    # Load dataset
    dataset = load_dataset(dataloader, data_files=data_files)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Tokenize dataset
    def generate_dataset_instances(instance):
        return {
            'options': f"{instance['opa']}</s>{instance['opb']}</s>{instance['opc']}</s>{instance['opd']}</s>{instance['ope']}",
            'question+options': f"{instance['question']}</s>{instance['opa']}</s>{instance['opb']}</s>{instance['opc']}</s>{instance['opd']}</s>{instance['ope']}",
            'full_instance': f"{instance['question']}</s>{instance['opa']}</s>{instance['opb']}</s>{instance['opc']}</s>{instance['opd']}</s>{instance['ope']}</s>{instance['answer_justification']}"
        }

    get_tokenized_length = lambda x: tokenizer(x, padding=False, truncation=False, return_token_type_ids=False, return_attention_mask=False, return_length=True).length

    def tokenize_function(examples):
        # Get the multiple input_ids for each instance combinations: question, options, question+options, full_instance, ...
        return {
            'question': get_tokenized_length(examples['question']),
            'options': get_tokenized_length(examples['options']),
            'question+options': get_tokenized_length(examples['question+options']),
            'full_instance': get_tokenized_length(examples['full_instance']),
            'answer_justification': get_tokenized_length(examples['answer_justification']),
            'opa': get_tokenized_length(examples['opa']),
            'opb': get_tokenized_length(examples['opb']),
            'opc': get_tokenized_length(examples['opc']),
            'opd': get_tokenized_length(examples['opd']),
            'ope': get_tokenized_length(examples['ope']),
        }

    tokenized_dataset_lengths = dataset.map(generate_dataset_instances)
    tokenized_dataset_lengths = tokenized_dataset_lengths.map(tokenize_function, batched=True)

    # Convert to numpy
    tokenized_dataset_lengths.set_format(type='numpy', columns=['question', 'answer_justification', 'opa', 'opb', 'opc', 'opd', 'ope', 'label', 'options', 'question+options', 'full_instance'])

    # Get a dataframe with the lengths

    lengths_df = pd.DataFrame(tokenized_dataset_lengths['train'])

    # Obtain Plots and statistics using Plotly
    # Different length distribution
    # Melt the dataframe
    lengths_df_melt = lengths_df.melt(value_vars=['question', 'answer_justification', 'opa', 'opb', 'opc', 'opd', 'ope', 'options', 'question+options', 'full_instance'], var_name='Section', value_name='Length')
    fig = px.box(
        lengths_df_melt, x='Length', y='Section',
        title='Length distribution of the different fields in the dataset',
        color='Section',
        category_orders={'Section': ['full_instance', 'question+options', 'question', 'options', 'opa', 'opb', 'opc', 'opd', 'ope', 'answer_justification']},
    )
    # Add a red line for the maximum length (512)
    fig.add_shape(
        type='line', line=dict(color='red', width=0.5), x0=512, x1=512, y0=-0.5, y1=9.5,
    )
    # Remove legend
    fig.update_layout(showlegend=False)

    # Save the plot
    fig.write_image(str(BASE_PATH / 'imgs' / f'length_distribution.{language}.{split}.png'))
    fig.write_html(str(BASE_PATH / 'imgs' / f'length_distribution.{language}.{split}.html'))

    # Label distribution plot
    # Map 0-4 to the corresponding labels: A, B, C, D, E
    lengths_df['label'] = lengths_df['label'].map({0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'})
    fig = px.histogram(
        lengths_df, x='label',
        title='Label distribution in the dataset',
        color='label',
        category_orders={'label': ['A', 'B', 'C', 'D', 'E']},
    )
    fig.update_layout(xaxis_title='Label', yaxis_title='Count')
    fig.write_image(str(BASE_PATH / 'imgs' / f'label_distribution.{language}.{split}.png'))
    fig.write_html(str(BASE_PATH / 'imgs' / f'label_distribution.{language}.{split}.html'))


if __name__ == '__main__':
    for lang in ['en', 'es']:
        # All splits
        check_lengths_casimedicos(
            model_path=str((BASE_PATH / 'models/eriberta_libre').absolute()),
            data_files=[
                str((BASE_PATH / f"data/casiMedicos/JSONL/{lang}.{_split}_casimedicos.jsonl").absolute())
                for _split in SPLITS
            ],
            dataloader=str((BASE_PATH / 'data/casiMedicos/casimedicos_data_loader.py').absolute()),
            split='All',
            language=lang,
        )

        # Train split
        # check_lengths_casimedicos(
        #     model_path=str((BASE_PATH / 'models/eriberta_libre').absolute()),
        #     data_files=str((BASE_PATH / f'data/casiMedicos/JSONL/{lang}.train_casimedicos.jsonl').absolute()),
        #     dataloader=str((BASE_PATH / 'data/casiMedicos/casimedicos_data_loader.py').absolute()),
        #     split='Train',
        #     language=lang,
        # )
