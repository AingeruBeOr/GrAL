Esto es para lo mismo que "4-1-fixing-error-v_old" pero utilizando el código de "3-2baseline_Paula_balanceado" y solo creando el nuevo dataset de prueba en [dataset.py](./dataset.py) que se llama FixingErrorDataset.

Cambios (comparando los archivos de "4-1-fixing-error-v_old" y "3-2baseline_Paula_balanceado"):

 - args.py: no cambia
 - models.py: no cambia
 - train.py: cambia:
    - WB_PROJECT = "tfg-MedMCQA-and-CasiMedicos"
    - MODELS_FOLDER = "/home/shared/esperimentuak/AingeruTFG/TFG/models/MedMCQA_and_CasiMedicos"
    - train_dataset = FixingErrorDataset(jsonl_path_casimedicos=args.train_csv, jsonl_path_medmcqa='../../data/MedMCQA-balanced/train.json', use_context=args.use_context)
 - dataset.py: cambia:
    - nuevo modelo FixingErrorDataset añadido

El mismo entrenamiento pero con el datset solo de casimedicos balanceado está en https://wandb.ai/abellido/tfg-baseline-Paula-synthetic/runs/eriberta_libre___dataen.train_casimedicos.jsonl___seqlen400___execTime2024-03-31-18-26?nw=nwuserabellido


## Resultados y conclusiones:

A la vista de que el único cambio en el entrenamiento es el dataset, los valores se asemejan mucho a los conseguidos con los mismos parámetros con solo casimédicos balanceados en "3-2baseline_Paula_balanceado" pero son ligeramente peores en términos de accuracy y loss.