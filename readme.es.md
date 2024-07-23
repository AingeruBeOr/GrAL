# Sistema bilingüe para resolver preguntas médicas: aportación de diferentes arquitecturas de modelos de lenguaje

[![Static Badge](https://img.shields.io/badge/%F0%9F%A4%97%20Huggingface-blue?style=for-the-badge)](https://huggingface.co/) ![Static Badge](https://img.shields.io/badge/PyTorch-F1A77D?style=for-the-badge&logo=pytorch) ![Static Badge](https://img.shields.io/badge/WANDB-black?style=for-the-badge&logo=weightsandbiases)


El presente trabajo se centra en la implementación y entrenamiento de un modelo de lenguaje bilingüe de Inteligencia Artificial capaz de resolver preguntas de exámenes médicos Médico Interno Residente (MIR) con un número variable de respuestas posibles, utilizando técnicas del estado del arte del  Procesamiento del Lenguaje Natural y el Aprendizaje Profundo.

Basándose en estudios previos en esta tarea, la experimentación de este trabajo cuenta con una sólida base de conocimiento para el entrenamiento de los modelos. A diferencia de previas investigaciones, los modelos generados en este trabajo reciben la información únicamente a través de los ejemplos de exámenes MIR sin consultar en fuentes de información externas.

Los modelos de lenguaje desarrollados en el presente trabajo son capaces de aprender las singularidades de las preguntas de los exámenes MIR. Para ello, se proponen diferentes arquitecturas de modelos que dan respuesta a preguntas de exámenes con un número variable de opciones posibles modificando las arquitecturas de investigaciones previas. 

Haciendo uso de conjuntos de exámenes en español e inglés, los modelos generados en este proyecto presentan una capacidad bilingüe para responder estos exámenes. Esto contrasta con la mayoría de modelos desarrollados hasta la fecha en este tipo de tareas, que utilizan únicamente el inglés.

## Arquitecturas

Se ha experimentado con dos arquitecturas de modelos de lenguaje, *baseline* y la arquitectura MedMCQA.

Aquitectura *baseline*:

![arquitectura baseline](baseline_arch.png)

Aquitectura MedMCQA:

![arquitectura MedMCQA](MedMCQA_arch.png)

## *Datasets*

Los dos *datasets* usados han sido:

- *Dataset* MedMCQA:
    - Sacado de [github.com/medmcqa/medmcqa](https://github.com/medmcqa/medmcqa). También disponible: [huggingface/medmcqa](https://huggingface.co/datasets/medmcqa)
- *Dataset* CasiMedicos: sacado de [github.com/ixa-ehu/antidote-casimedicos](https://github.com/ixa-ehu/antidote-casimedicos)

## Pre-trained model

El modelo pre-entrenado a usar es [EriBERTa: A Bilingual Pre-Trained Language Model for Clinical Natural Language Processing
](https://arxiv.org/abs/2306.07373). Disponible en [HiTZ/EriBERTa-base](https://huggingface.co/HiTZ/EriBERTa-base).

## Project structure

Los experimentos están listado en el directorio [src](src). Cada uno tiene su propio directorio con su correspondiente archivo readme.

## Otras dependencias

![Static Badge](https://img.shields.io/badge/TQDM-grey?style=for-the-badge&logo=tqdm) ![Static Badge](https://img.shields.io/badge/matplotlib-grey?style=for-the-badge) ![Static Badge](https://img.shields.io/badge/pandas-grey?style=for-the-badge&logo=pandas) ![Static Badge](https://img.shields.io/badge/pyplot-grey?style=for-the-badge&logo=plotly) ![Static Badge](https://img.shields.io/badge/scikit--learn-grey?style=for-the-badge&logo=scikitlearn) ![Static Badge](https://img.shields.io/badge/numpy-grey?style=for-the-badge&logo=numpy) 