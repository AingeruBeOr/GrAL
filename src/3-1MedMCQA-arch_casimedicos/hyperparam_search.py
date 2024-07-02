from args import Arguments
from model import MCQAModel
from dataset import CasiMedicosDatasetBalanced
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import torch,os
import wandb

'''
Adaptado de:
 - https://github.com/paulaonta/medmcqa/blob/7e9406e80e8118eeac7ff6ae4596de61ed003930/train.py
 - https://github.com/medmcqa/medmcqa/blob/main/train.py

 TODO is not finished yet
'''

# Set up Weights & Biases
os.environ["WANDB_START_METHOD"] = "thread"
os.environ["WANDB_LOG_MODEL"] = "false" # do not upload the model to wandb

DATASET_FOLDER = "../../data/casiMedicos-balanced/JSONL"
WB_PROJECT = "tfg-baseline-Paula"
MODELS_FOLDER = "/home/shared/esperimentuak/AingeruTFG/TFG/models/baseline_Paula"
PRETRAINED_MODEL = "/home/shared/esperimentuak/AingeruTFG/TFG/models/eriberta_libre/"

def train(gpu,
          args,
          # exp_dataset_folder, # TODO no se utiliza??? (raro), se podría borrar
          #experiment_name,
          models_folder,
          #version
          ):
    pl.seed_everything(42)

    torch.cuda.init()
    print("Cuda is available? " + str(torch.cuda.is_available()))
    print("Available devices? " + str(torch.cuda.device_count()))

    # Set up loggers
    wb = WandbLogger(project=WB_PROJECT)

    # Load train, test and val datasets
    train_dataset = CasiMedicosDatasetBalanced(args.train_csv,args.use_context)
    test_dataset = CasiMedicosDatasetBalanced(args.test_csv,args.use_context)
    val_dataset = CasiMedicosDatasetBalanced(args.dev_csv,args.use_context)

    early_stopping_callback = pl.callbacks.EarlyStopping(monitor='val_acc',
                                    min_delta=0.00,
                                    patience=args.early_stopping_patience,
                                    verbose=True,
                                    mode='max')


    # Load pretrained model
    mcqaModel = MCQAModel(model_name_or_path=args.pretrained_model_name,
                      args=args.__dict__)
    
    mcqaModel.prepare_dataset(train_dataset=train_dataset,
                              test_dataset=test_dataset,
                              val_dataset=val_dataset)

    trainer = Trainer(
        accelerator='gpu',
        #devices=gpu,
        gpus=gpu, # TODO deprecated 
        #distributed_backend='ddp' if not isinstance(gpu,list) else None, #deprecated
        strategy="ddp" if not isinstance(gpu, list) else None,
        logger=wb,
        callbacks= early_stopping_callback,
        #resume_from_checkpoint=pretrained_model,
        max_epochs=args.num_epochs,                          # EPOCHS
        accumulate_grad_batches=args.accumulate_grad_batches # Gradient accumulation
    )
    
    # Start training
    trainer.fit(mcqaModel)
    print(f"Training completed")

if __name__ == "__main__":
    # TODO argument parsing (if neccessary)

    # Initialize wandb
    wandb.init(project=WB_PROJECT)

    args = Arguments(train_csv=os.path.join(DATASET_FOLDER, "en.train_casimedicos.jsonl"),
                    test_csv=os.path.join(DATASET_FOLDER, "en.test_casimedicos.jsonl"),
                    dev_csv=os.path.join(DATASET_FOLDER, "en.dev_casimedicos.jsonl"),
                    accumulate_grad_batches=wandb.config.accumulate_grad_batches,
                    learning_rate=wandb.config.learning_rate,
                    num_epochs=wandb.config.num_epochs,
                    pretrained_model_name=PRETRAINED_MODEL,
                    use_context=False)
    
    print(f"Training started for model - {args.pretrained_model_name} variant - {DATASET_FOLDER} use_context - {str(args.use_context)}")
    
    train(gpu=args.gpu,
        args=args,
        #exp_dataset_folder=DATASET_FOLDER,
        #experiment_name=exp_name,
        models_folder=MODELS_FOLDER,
        #version=exp_name
        )
    