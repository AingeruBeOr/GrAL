# MedMCQA-architecture and MedMCQA-dataset+CasiMedicos-dataset

This model is finetuned using:

- Dataset: CasiMedicos (english) balanced + MedMCQA
- Architecture: MedMCQA 

The trainings are using the next splits:

- Dataset:
    - train split: MedMCQA.train (balanced) + CasiMédicos.train (balanced). Number of instances: 182.822\*4 + 404\*5 = 733308
    - dev split: CasiMédicos.train (balanced)
    - test split: CasiMédicos.train (balanced)

Outstanding tasks:
- Generate a MedMCQA balanced dataset
- Train the model with both available datasets