from args import Arguments
from model import MCQAModel
from dataset import CasiMedicosDatasetBalanced
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers import CSVLogger
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import torch, os
import json
from tqdm import tqdm
import time, argparse, datetime

'''
Adaptado de:
 - https://github.com/paulaonta/medmcqa/blob/7e9406e80e8118eeac7ff6ae4596de61ed003930/train.py
 - https://github.com/medmcqa/medmcqa/blob/main/train.py
'''


def train(gpu, args:Arguments, experiment_name, version):

    # Afecta a iniciazliación de pesos, aleatoriedad en el dataset, etc.
    pl.seed_everything(args.seed)

    torch.cuda.init()
    print("Cuda is available? " + str(torch.cuda.is_available()))
    print("Available devices? " + str(torch.cuda.device_count()))

    # Set up experiment folder
    EXPERIMENT_FOLDER = os.path.join(args.models_folder, experiment_name)
    os.makedirs(EXPERIMENT_FOLDER, exist_ok=True)
    checkpoints_dir = os.path.join(EXPERIMENT_FOLDER, 'checkpoints')
    print(f"Checkpoints dir: {checkpoints_dir}")

    # Set up loggers
    wb = WandbLogger(project=args.wandb_project_name, name=experiment_name, version=version)
    csv_log = CSVLogger(args.models_folder, name=experiment_name, version=version)

    # Load train, test and val datasets
    train_dataset = CasiMedicosDatasetBalanced(args.train_csv, args.use_context)
    test_dataset = CasiMedicosDatasetBalanced(args.test_csv, args.use_context)
    val_dataset = CasiMedicosDatasetBalanced(args.dev_csv, args.use_context)

    # Early stopping callback
    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor=args.metric_for_best_model,
        min_delta=0.00,
        patience=args.early_stopping_patience,
        verbose=True,
        mode=args.metric_for_best_model_mode
    )

    # Model checkpoint callback. Reference: https://lightning.ai/docs/pytorch/1.9.0/api/pytorch_lightning.callbacks.ModelCheckpoint.html#pytorch_lightning.callbacks.ModelCheckpoint
    cp_callback = pl.callbacks.ModelCheckpoint(
        monitor=args.metric_for_best_model,
        dirpath=checkpoints_dir,
        filename='{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}',
        save_top_k=args.n_checkpoints_to_save,
        save_weights_only=False, # if False, optimizer states, lr-scheduler states, etc are added in the checkpoint too
        mode=args.metric_for_best_model_mode
    )

    # Load pretrained model
    mcqaModel = MCQAModel(
        model_name_or_path=args.pretrained_model,
        args=args.__dict__
    )
    
    mcqaModel.prepare_dataset(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        val_dataset=val_dataset
    )

    trainer = Trainer(
        accelerator='gpu',
        #devices=gpu,
        gpus=gpu,
        strategy="ddp" if not isinstance(gpu, list) else None,
        logger=[wb, csv_log],
        callbacks= [early_stopping_callback, cp_callback],
        #resume_from_checkpoint=pretrained_model,
        max_epochs=args.num_epochs,                           # EPOCHS
        accumulate_grad_batches=args.accumulate_grad_batches, # Gradient accumulation
        val_check_interval=args.val_check_interval,           # Validation check interval (how many batches to consider before running validation)
        log_every_n_steps=args.log_every_n_steps,             # Log every n steps
    )
    
    # Start training
    trainer.fit(mcqaModel)
    print(f"Training completed")

    ckpt = [f for f in os.listdir(checkpoints_dir) if f.endswith('.ckpt')]
    print(f"Checkpoints list {ckpt}")



    # --- Inference ---

    # Load best model at training to use it for inference on the test set
    best_checkpoint_model_path = os.path.join(checkpoints_dir, ckpt[0]) # TODO el 0 seguro que es el best??
    inference_model = MCQAModel.load_from_checkpoint(best_checkpoint_model_path)
    inference_model.prepare_dataset(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        val_dataset=val_dataset
    )
    inference_model = inference_model.to("cuda")
    inference_model = inference_model.eval()
    
    # Test the model on the test set using the best model
    inference_datasets = [
        os.path.join(args.dataset_folder, "es.dev_casimedicos.jsonl"),
        os.path.join(args.dataset_folder, "es.test_casimedicos.jsonl"),
        os.path.join(args.dataset_folder, "en.dev_casimedicos.jsonl"),
        os.path.join(args.dataset_folder, "en.test_casimedicos.jsonl")
    ]
    for inference_dataset in inference_datasets:
        inference_model.test_dataset = CasiMedicosDatasetBalanced(inference_dataset, args.use_context)
        test_results = trainer.test(model=inference_model, dataloaders=inference_model.test_dataloader())
        inference_type_metric_name = os.path.basename(inference_dataset).split('_')[0] + '_inference_accuracy'
        wb.log_metrics({inference_type_metric_name: test_results[0]})
        csv_log.log_metrics({inference_type_metric_name: test_results[0]})
    
    # Free up memory
    del mcqaModel
    del inference_model
    del trainer
    torch.cuda.empty_cache()
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    #parser.add_argument("--model", default="eriberta_libre", help="name of the model")
    #parser.add_argument("--dataset_folder_name", default="content/medmcqa_data/", help="dataset folder")
    #parser.add_argument("--use_context", default=False, action='store_true', help="mention this flag to use_context")
    cmd_args = parser.parse_args()
    #model = cmd_args.model

    # Get the arguments (hiperparameters, paths, training and model specs)
    args = Arguments()

    model_name = os.path.basename(args.pretrained_model)
    print(f"Training started for model - {model_name} variant - {args.dataset_folder} use_context - {str(args.use_context)}")

    # Set up Weights & Biases
    os.environ["WANDB_START_METHOD"] = args.wandb_start_method
    os.environ["WANDB_LOG_MODEL"] = args.wandb_log_model
    
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d-%H-%M")
    exp_name = str(formatted_datetime).replace("/","_")
    train(gpu=args.gpu, args=args, experiment_name=exp_name, version=exp_name)
    
    time.sleep(60) # TODO, esto para qué sirve?