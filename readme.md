#  Arquitecturas sobre MedMCQA

## Dataset

Recogido de [https://github.com/medmcqa/medmcqa](https://github.com/medmcqa/medmcqa)

También disponible en [huggingface/medmcqa](https://huggingface.co/datasets/medmcqa)

## Model

The model to use is [EriBERTa: A Bilingual Pre-Trained Language Model for Clinical Natural Language Processing
](https://arxiv.org/abs/2306.07373)

## Estructura del proyecto

```
.
├── data
│   └── MedMCQA/
├── models
│   └── eriberta_libre/
├── readme.md
├── requirements.txt
└── src
    ├── datasetStats.ipynb
    ├── datasetStats.py
    ├── loadDataset.py
    ├── train.py
    └── variables.py
```