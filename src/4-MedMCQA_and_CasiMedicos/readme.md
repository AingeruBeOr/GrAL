Este experimento consiste en:

- Dataset:
    - train split: MedMCQA.train (balanceado) + CasiMédicos.train (balanceado). Número de instancias: 182.822\*4 + 404\*5 = 733308
    - dev split: CasiMédicos.train (balanceado)
    - test split: CasiMédicos.train (balanceado)
- Arquitectura: MedMCQA (Paula)


Tareas destacadas:
- Generar MedMCQA balanceado
- Probar con ambos datasets

Problemas encontrados duarante el entrenamiento (toda la info en mi Notion):
- El val_acc no mejoraba durante el entrenamiento.
    - Había un error en la implementación, el CrossEntropy se debería haber generado con la suma de los diferentes batches y no con la media que es lo que está por defecto.