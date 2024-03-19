from dataclasses import dataclass

'''
Adaptado de: https://github.com/paulaonta/medmcqa/blob/7e9406e80e8118eeac7ff6ae4596de61ed003930/conf/args.py
'''

@dataclass
class Arguments:
    train_csv:str
    test_csv:str 
    dev_csv:str
    incorrect_ans:int
    batch_size:int = 4
    max_len:int = 192
    checkpoint_batch_size:int = 32
    print_freq:int = 100
    pretrained_model_name:str = "bert-base-uncased"
    learning_rate:float = 2e-4
    hidden_dropout_prob:float =0.4
    hidden_size:int=768#512#768
    num_epochs:int = 5
    num_choices:int = 4
    device:str='cuda'
    gpu='0,1'
    use_context:bool=True