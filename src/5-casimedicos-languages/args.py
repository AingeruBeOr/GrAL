from dataclasses import dataclass
import os

'''
Adaptado de: https://github.com/paulaonta/medmcqa/blob/7e9406e80e8118eeac7ff6ae4596de61ed003930/conf/args.py

Pytorch lightning training arguments: https://lightning.ai/docs/pytorch/1.9.0/api/pytorch_lightning.trainer.trainer.Trainer.html
'''

@dataclass
class Arguments:

    # Wandb arguments
    wandb_start_method:str = "thread"                       # Wandb start method
    wandb_log_model:str = "false"                           # Log model to wandb
    wandb_project_name:str = "tfg-CasiMedicos-languages"    # Wandb project name

    # Paths
    dataset_folder:str = "../../data/casiMedicos-balanced/JSONL"                                  # Dataset folder
    models_folder:str = "/home/shared/esperimentuak/AingeruTFG/TFG/models/CasiMedicos-languages"  # Models folder
    pretrained_model:str = "/home/shared/esperimentuak/AingeruTFG/TFG/models/eriberta_libre"      # Pretrained model

    # Training arguments
    train_csv:str = os.path.join(dataset_folder, "en.train_casimedicos.jsonl")   # Path to the training CSV file # TODO el nuestro es JSONL
    test_csv:str = os.path.join(dataset_folder, "en.test_casimedicos.jsonl")           # Path to the test CSV file # TODO el nuestro es JSONL
    dev_csv:str = os.path.join(dataset_folder, "en.dev_casimedicos.jsonl")       # Path to the dev CSV file # TODO el nuestro es JSONL
    max_len:int = 400                                       # Max length of the maximum possible input (maximum value: 512)
    num_epochs:int = 30                                     # EPOCHS
    batch_size:int = 3                                      # Batch size
    accumulate_grad_batches:int = 3                         # Gradient accumulation
    learning_rate:float = 5e-5                              # Learning rate
    val_check_interval:int | float = 1.0                    # Validation check interval (how many batches to consider before running validation). It has to be greater than the steps before backpropagation. Caution! It can be tricky to set this parameter.
    log_every_n_steps:int = 1                               # Log every n steps
    #checkpoint_batch_size:int = 32
    early_stopping_patience:int = 20                        # Early stopping patience
    print_freq:int = 100
    metric_for_best_model:str = 'val_acc'                   # Metric for best model
    metric_for_best_model_mode:str = 'max'                  # Metric for best model mode
    n_checkpoints_to_save:str = 1                           # Number of checkpoints to save # TODO if > 1, change code when loading best model
    hidden_dropout_prob:float = 0.4                         # Dropout
    seed:int = 42                                           # Seed for reproducibility. Afecta a iniciazliación de pesos, aleatoriedad en el dataset, etc.
    device:str='cuda'
    gpu=0                                                   # '0,1' # a lo que esté establecido en CUDA_VISIBLE_DEVICES. Aunque si está establecido a 1, como solo es visible la 1 también hay que poner 0. Si estuvieran en CUDA_VISIBLE_DEVICES las dos, podríamos elegir

    # Model specs
    hidden_size:int=768                                     # Tamaño de la capa de clasificación. Como solo hay una capa, es el tamaño de la representación del primer token

    # Dataset specs
    num_choices:int = 5                                     # Opciones posibles en las respuestas
    use_context:bool=False                                  # Si se usa el contexto en el dataset