from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from loadDataset import MedMCQA_Dataset, load_dataset_as_dict

LR = 2e-5
EPOCHS = 1
BATCH_SIZE = 16
TOKENIZER = ""
MODEL = "" # use this to finetune the language model

# https://huggingface.co/docs/transformers/main/en/tasks/sequence_classification


## 1. LOAD DATASET ##
#train_dataset = MedMCQA_Dataset("../data/train.json")
train_dataset = load_dataset_as_dict("../data/train.json")
dev_dataset = load_dataset_as_dict("../data/dev.json")

## 2. TOKENIZATION ##

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, use_fast=True)

encoded_train = tokenizer(train_dataset['inputs']) # TODO: hace falta truncar??
encoded_dev = tokenizer(dev_dataset['inputs']) # TODO: hace falta truncar??

## 3. FINE-TUNE MODEL ##
# Load a torch.utils.data.Dataset object (required in Trainer)
encoded_train_dataset = MedMCQA_Dataset(encoded_train, train_dataset['labels'])
encoded_dev_dataset = MedMCQA_Dataset(encoded_dev, dev_dataset['labels'])

# Load model
model = AutoModelForSequenceClassification.from_pretrained(MODEL) # Transformer with default classification head

# SET TRAINING HYPERPARAMETERS #
training_args = TrainingArguments(
    output_dir='',                              # output directory
    evaluation_strategy='steps',                # when to evaluate
    num_train_epochs=EPOCHS,                    # total number of training epochs
    per_device_train_batch_size=BATCH_SIZE,     # batch size per device during training
    per_device_eval_batch_size=BATCH_SIZE,      # batch size for evaluation
    warmup_steps=100,                           # number of warmup steps for learning rate scheduler
    #weight_decay=
    logging_dir='./logs',                       # directory for storing logs
    logging_steps=10,                           # when to print log (and to evaluate if evaluation_strategy = 'steps')
    load_best_model_at_end=True                 # load or not best model at the end
    #use_mps_device=True                        # para que use la GPU del MAC
)

# SET TRAINER: MODEL TO TRAIN, TRAINING ARGUMENTS, DATSET SPLITS...
trainer = Trainer(
    model=model,
    args=training_args,                   # training arguments, defined above
    train_dataset=encoded_train_dataset,  # training dataset. Format: torch.utils.data.Dataset
    eval_dataset=encoded_dev_dataset      # evaluation dataset. Format: torch.utils.data.Dataset
)

# FINE-TUNE MODEL ##
trainer.train()
