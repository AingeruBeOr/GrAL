from args import Arguments
from model import MCQAModel
from dataset import MedMCQAAndCasiMedicosDataset, CasiMedicosDatasetBalanced
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

# Set up Weights & Biases
os.environ["WANDB_START_METHOD"] = "thread"
os.environ["WANDB_LOG_MODEL"] = "false" # do not upload the model to wandb

DATASET_FOLDER = "../../data/casiMedicos-balanced/JSONL"
WB_PROJECT = "tfg-MedMCQA-and-CasiMedicos"
MODELS_FOLDER = "/home/shared/esperimentuak/AingeruTFG/TFG/models/MedMCQA_and_CasiMedicos"
PRETRAINED_MODEL = "/home/shared/esperimentuak/AingeruTFG/TFG/models/eriberta_libre/"


def train(gpu, args, experiment_name, version):

    # Afecta a iniciazliación de pesos, aleatoriedad en el dataset, etc.
    pl.seed_everything(args.seed)

    torch.cuda.init()
    print("Cuda is available? " + str(torch.cuda.is_available()))
    print("Available devices? " + str(torch.cuda.device_count()))

    # Set up experiment folder
    EXPERIMENT_FOLDER = os.path.join(MODELS_FOLDER, experiment_name)
    os.makedirs(EXPERIMENT_FOLDER, exist_ok=True)
    checkpoints_dir = os.path.join(EXPERIMENT_FOLDER, 'checkpoints')
    print(f"Checkpoints dir: {checkpoints_dir}")

    # Set up loggers
    wb = WandbLogger(project=WB_PROJECT, name=experiment_name, version=version)
    csv_log = CSVLogger(MODELS_FOLDER, name=experiment_name, version=version)

    # Load train, test and val datasets
    train_dataset = MedMCQAAndCasiMedicosDataset(
        casimedicos_dataset_path=args.train_csv, 
        medmcqa_dataset_path='../../data/MedMCQA-balanced/train.json',
        use_context=args.use_context
    )
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
        model_name_or_path=args.pretrained_model_name,
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
        max_epochs=args.num_epochs,                          # EPOCHS
        accumulate_grad_batches=args.accumulate_grad_batches # Gradient accumulation
    )
    
    # Start training
    trainer.fit(mcqaModel)
    print(f"Training completed")

    ckpt = [f for f in os.listdir(checkpoints_dir) if f.endswith('.ckpt')]
    print(f"Checkpoints list {ckpt}")

    # Load best model at training to use it for inference on the test set
    best_checkpoint_model_path = os.path.join(checkpoints_dir, ckpt[0]) # TODO el 0 seguro que es el best??
    inference_model = MCQAModel.load_from_checkpoint(best_checkpoint_model_path)
    inference_model = inference_model.to("cuda")
    inference_model = inference_model.eval()
    
    # Test the model on the test set using the best model
    test_results = trainer.test(ckpt_path=best_checkpoint_model_path) # TODO, arreglar los argumentos de test
    wb.log_metrics(test_results[0])
    csv_log.log_metrics(test_results[0])
    
    # Persist test dataset predictions
    test_set = []
    with open(args.test_csv, 'r') as file:
      test_set = [json.loads(row) for row in file]
    predictions = run_inference(inference_model, mcqaModel.test_dataloader(), args)
    for instance, prediction in zip(test_set, predictions):
      instance['prediction'] = prediction.item() # numpy.int64 to int to be able to serialize to json
    with open(os.path.join(EXPERIMENT_FOLDER, 'test_predictions.jsonl'), 'w') as file:
        for instance in test_set:
            file.write(json.dumps(instance) + '\n')
    print(f"Test predictions written to {os.path.join(EXPERIMENT_FOLDER, 'test_predictions.jsonl')}")

    # Persist val dataset predictions
    val_set = []
    with open(args.dev_csv, 'r') as file:
        val_set = [json.loads(row) for row in file]
    predictions = run_inference(inference_model, mcqaModel.val_dataloader(), args)
    for instance, prediction in zip(val_set, predictions):
        instance['prediction'] = prediction.item() # numpy.int64 to int to be able to serialize to json
    with open(os.path.join(EXPERIMENT_FOLDER, 'dev_predictions.jsonl'), 'w') as file:
        for instance in val_set:
            file.write(json.dumps(instance) + '\n')
    print(f"Val predictions written to {os.path.join(EXPERIMENT_FOLDER, 'dev_predictions.jsonl')}")

    # Free up memory
    del mcqaModel
    del inference_model
    del trainer
    torch.cuda.empty_cache()
    

def run_inference(model,dataloader,args):
    predictions = []
    for idx,(inputs,labels) in tqdm(enumerate(dataloader)):
        for key in inputs.keys():
            inputs[key] = inputs[key].to(args.device)
        with torch.no_grad(): # Don't store the gradients. As we are not doing backpropagation in inference, there is no need to save them
            outputs = model(**inputs)
        prediction_idxs = torch.argmax(outputs,axis=1).cpu().detach().numpy()
        predictions.extend(list(prediction_idxs))
    return predictions
        


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="eriberta_libre", help="name of the model")
    #parser.add_argument("--dataset_folder_name", default="content/medmcqa_data/", help="dataset folder")
    parser.add_argument("--use_context", default=False, action='store_true', help="mention this flag to use_context")
    cmd_args = parser.parse_args()

    model = cmd_args.model
    print(f"Training started for model - {model} variant - {DATASET_FOLDER} use_context - {str(cmd_args.use_context)}")

    args = Arguments(
        train_csv=os.path.join(DATASET_FOLDER, "en.train_casimedicos.jsonl"),
        test_csv=os.path.join(DATASET_FOLDER, "en.test_casimedicos.jsonl"),
        dev_csv=os.path.join(DATASET_FOLDER, "en.dev_casimedicos.jsonl"),
        pretrained_model_name=PRETRAINED_MODEL,
        use_context=cmd_args.use_context
    )
    
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d-%H-%M")
    exp_name = f"{model}___data{os.path.basename(args.train_csv)}___seqlen{str(args.max_len)}___execTime{str(formatted_datetime)}".replace("/","_")

    train(gpu=args.gpu, args=args, experiment_name=exp_name, version=exp_name)
    
    time.sleep(60) # TODO, esto para qué sirve?