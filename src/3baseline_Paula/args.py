from dataclasses import dataclass

'''
Adaptado de: https://github.com/paulaonta/medmcqa/blob/7e9406e80e8118eeac7ff6ae4596de61ed003930/conf/args.py
'''

@dataclass
class Arguments:
    train_csv:str                                           # Path to the training CSV file # TODO el nuestro es JSONL
    test_csv:str                                            # Path to the test CSV file # TODO el nuestro es JSONL
    dev_csv:str                                             # Path to the dev CSV file # TODO el nuestro es JSONL
    incorrect_ans:int
    batch_size:int = 4                                      # Batch size
    max_len:int = 192
    checkpoint_batch_size:int = 32
    print_freq:int = 100
    pretrained_model_name:str = "bert-base-uncased"         # Pretrained model
    learning_rate:float = 2e-4                              # Learning rate
    hidden_dropout_prob:float =0.4                          # Dropout
    hidden_size:int=768#512#768                             # Tamaño de la capa oculta
    num_epochs:int = 5                                      # EPOCHS
    num_choices:int = 5                                     # Opciones posibles en las respuestas
    device:str='cuda'
    gpu=0                                                   # '0,1' # a lo que esté establecido en CUDA_VISIBLE_DEVICES. Aunque si está establecido a 1, como solo es visible la 1 también hay que poner 0. Si estuvieran en CUDA_VISIBLE_DEVICES las dos, podríamos elegir
    use_context:bool=True                                   # Si se usa el contexto en el dataset