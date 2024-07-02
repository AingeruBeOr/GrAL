# Sistema automático bilingüe para resolver exámenes MIR

[![Static Badge](https://img.shields.io/badge/%F0%9F%A4%97%20Huggingface-blue?style=for-the-badge)](https://huggingface.co/) ![Static Badge](https://img.shields.io/badge/PyTorch-F1A77D?style=for-the-badge&logo=pytorch) ![Static Badge](https://img.shields.io/badge/WANDB-black?style=for-the-badge&logo=weightsandbiases)




El presente trabajo se centra en la implementación y entrenamiento de un modelo bilingüe de Inteligencia Artificial (IA) capaz de resolver preguntas de exámenes MIR (Médico Interno Residente) con un número variable de respuestas posibles, utilizando técnicas del estado del arte del  Procesamiento del Lenguaje Natural (PLN) y el aprendizaje profundo.

Basándose en estudios previos en esta tarea, la experimentación de este trabajo cuenta con una sólida base de conocimiento para el entrenamiento de los modelos. A diferencia de previas investigaciones, los modelos generados en este trabajo reciben la información únicamente a través de los ejemplos de exámenes MIR sin consultar en fuentes de información externas.

Los modelos desarrollados en el presente trabajo son capaces de aprender las singularidades de las preguntas de los exámenes MIR. Para ello, se proponen diferentes arquitecturas de modelos que dan respuesta a preguntas de exámenes con un número variable de opciones posibles modificando las arquitecturas de investigaciones previas. 

Haciendo uso de conjuntos de exámenes en español e inglés, los modelos generados presentan una capacidad bilingüe para responder estos exámenes. Esto contrasta con la mayoría de modelos desarrollados hasta la fecha en este tipo de tareas, que utilizan únicamente el inglés.

## Arquitecturas

## Otras dependencias

![Static Badge](https://img.shields.io/badge/TQDM-grey?style=for-the-badge&logo=tqdm) ![Static Badge](https://img.shields.io/badge/matplotlib-grey?style=for-the-badge) ![Static Badge](https://img.shields.io/badge/pandas-grey?style=for-the-badge&logo=pandas) ![Static Badge](https://img.shields.io/badge/pyplot-grey?style=for-the-badge&logo=plotly) ![Static Badge](https://img.shields.io/badge/scikit--learn-grey?style=for-the-badge&logo=scikitlearn) ![Static Badge](https://img.shields.io/badge/numpy-grey?style=for-the-badge&logo=numpy) 