from args import Arguments
from model import MCQAModel
from dataset import CasiMedicosDatasetBalanced
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers import CSVLogger
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import torch,os
import pandas as pd
import json
from tqdm import tqdm
# from multiprocessing import Process # TODO no se utiliza, se podría borrar
import time,argparse,datetime

'''
Adaptado de:
 - https://github.com/paulaonta/medmcqa/blob/7e9406e80e8118eeac7ff6ae4596de61ed003930/train.py
 - https://github.com/medmcqa/medmcqa/blob/main/train.py
'''

# Set up Weights & Biases
os.environ["WANDB_START_METHOD"] = "thread"
os.environ["WANDB_LOG_MODEL"] = "false" # do not upload the model to wandb

DATASET_FOLDER = "../../data/casiMedicos-balanced/JSONL"
WB_PROJECT = "tfg-baseline-Paula-synthetic"
MODELS_FOLDER = "/home/shared/esperimentuak/AingeruTFG/TFG/models/baseline_Paula_synthetic"
PRETRAINED_MODEL = "/home/shared/esperimentuak/AingeruTFG/TFG/models/eriberta_libre/"

def train(gpu,
          args,
          # exp_dataset_folder, # TODO no se utiliza??? (raro), se podría borrar
          experiment_name,
          models_folder,
          version):
    #experiment_name = "bert-base-uncased@@@@@@use_contextFalse@@@daata._content_medmcqa_data_train_MEDMCQA_orig.csv@@@seqlen192"
    #pretrained_model = "./4ANSONLY_MIR_bert-base-uncased@@@@@@use_contextFalse@@@data._content_medmcqa_data_train_MEDMCQA_orig.csv@@@seqlen192/4ANSONLY_MIR_bert-base-uncased@@@@@@use_contextFalse@@@data._content_medmcqa_data_train_MEDMCQA_orig.csv@@@seqlen192-epoch=02-val_loss=1.33-val_acc=0.34.ckpt"

    pl.seed_everything(42)

    torch.cuda.init()
    print("Cuda is available? " + str(torch.cuda.is_available()))
    print("Available devices? " + str(torch.cuda.device_count()))

    # Set up experiment folder
    EXPERIMENT_FOLDER = os.path.join(models_folder, experiment_name)
    os.makedirs(EXPERIMENT_FOLDER,exist_ok=True)
    experiment_string = experiment_name+'-{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}'

    # Set up loggers
    wb = WandbLogger(project=WB_PROJECT, name=experiment_name, version=version)
    csv_log = CSVLogger(models_folder, name=experiment_name, version=version)

    # Load train, test and val datasets
    train_dataset = CasiMedicosDatasetBalanced(args.train_csv,args.use_context)
    test_dataset = CasiMedicosDatasetBalanced(args.test_csv,args.use_context)
    val_dataset = CasiMedicosDatasetBalanced(args.dev_csv,args.use_context)

    early_stopping_callback = pl.callbacks.EarlyStopping(monitor='val_acc',
                                    min_delta=0.00,
                                    patience=args.early_stopping_patience,
                                    verbose=True,
                                    mode='max')

    cp_callback = pl.callbacks.ModelCheckpoint(monitor='val_acc',
                                               #filepath=os.path.join(EXPERIMENT_FOLDER,experiment_string), # deprecated
                                               dirpath=os.path.join(EXPERIMENT_FOLDER,experiment_string),
                                               save_top_k=1,
                                               save_weights_only=False,
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
        gpus=gpu,
        #distributed_backend='ddp' if not isinstance(gpu,list) else None, #deprecated
        strategy="ddp" if not isinstance(gpu, list) else None,
        logger=[wb,csv_log],
        callbacks= [early_stopping_callback,cp_callback],
        #resume_from_checkpoint=pretrained_model,
        max_epochs=args.num_epochs,                          # EPOCHS
        accumulate_grad_batches=args.accumulate_grad_batches # Gradient accumulation
    )
    
    # Start training
    trainer.fit(mcqaModel)
    print(f"Training completed")

    checkpoints_dir = os.path.join(EXPERIMENT_FOLDER,experiment_string)
    print(f"Checkpoints dir: {checkpoints_dir}")
    ckpt = [f for f in os.listdir(checkpoints_dir) if f.endswith('.ckpt')]
    print(f"Checkpoints list {ckpt}")

    # Load best model at training to use it for inference on the test set
    inference_model = MCQAModel.load_from_checkpoint(os.path.join(checkpoints_dir,ckpt[0]))
    inference_model = inference_model.to("cuda")
    inference_model = inference_model.eval()
    
    # Test the model on the test set
    test_results = trainer.test(ckpt_path=os.path.join(checkpoints_dir,ckpt[0]))
    wb.log_metrics(test_results[0])
    csv_log.log_metrics(test_results[0])
    
    # Persist test dataset predictions
    #test_df = pd.read_csv(args.test_csv)
    test_set = []
    with open(args.test_csv, 'r') as file:
      test_set = [json.loads(row) for row in file]
    predictions = run_inference(inference_model,mcqaModel.test_dataloader(),args)
    for instance, prediction in zip(test_set, predictions):
      instance['prediction'] = prediction.item() # numpy.int64 to int to be able to serialize to json
    #test_df.loc[:,"predictions"] = [pred for pred in run_inference(inference_model,mcqaModel.test_dataloader(),args)]
    with open(os.path.join(EXPERIMENT_FOLDER, 'test_predictions.jsonl'), 'w') as file:
        for instance in test_set:
            file.write(json.dumps(instance) + '\n')
    #test_df.to_csv(os.path.join(EXPERIMENT_FOLDER,"test_results.csv"),index=False)
    print(f"Test predictions written to {os.path.join(EXPERIMENT_FOLDER, 'test_predictions.jsonl')}")

    #val_df = pd.read_csv(args.dev_csv)
    val_set = []
    with open(args.dev_csv, 'r') as file:
        val_set = [json.loads(row) for row in file]
    predictions = run_inference(inference_model,mcqaModel.val_dataloader(),args)
    for instance, prediction in zip(val_set, predictions):
        instance['prediction'] = prediction.item() # numpy.int64 to int to be able to serialize to json
    #val_df.loc[:,"predictions"] = [pred for pred in run_inference(inference_model,mcqaModel.val_dataloader(),args)]
    with open(os.path.join(EXPERIMENT_FOLDER, 'dev_predictions.jsonl'), 'w') as file:
        for instance in val_set:
            file.write(json.dumps(instance) + '\n')
    #val_df.to_csv(os.path.join(EXPERIMENT_FOLDER,"dev_results.csv"),index=False)
    print(f"Val predictions written to {os.path.join(EXPERIMENT_FOLDER, 'dev_predictions.jsonl')}")

    del mcqaModel
    del inference_model
    del trainer
    torch.cuda.empty_cache()
    
def run_inference(model,dataloader,args):
    predictions = []
    for idx,(inputs,labels) in tqdm(enumerate(dataloader)):
        # batch_size = len(labels) # TODO, no se usa, se podría borrar
        for key in inputs.keys():
            inputs[key] = inputs[key].to(args.device)
        with torch.no_grad(): # Don't store the gradients. As we are not doing backpropagation in inference, there is no need to save them
            outputs = model(**inputs)
        prediction_idxs = torch.argmax(outputs,axis=1).cpu().detach().numpy()
        predictions.extend(list(prediction_idxs))
    return predictions
        


if __name__ == "__main__":

    models = ["allenai/scibert_scivocab_uncased","bert-base-uncased"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="eriberta_libre", help="name of the model")
    #parser.add_argument("--dataset_folder_name", default="content/medmcqa_data/", help="dataset folder")
    parser.add_argument("--use_context", default=False, action='store_true', help="mention this flag to use_context")
    cmd_args = parser.parse_args()

    #exp_dataset_folder = os.path.join(EXPERIMENT_DATASET_FOLDER, cmd_args.dataset_folder_name)
    model = cmd_args.model
    print(f"Training started for model - {model} variant - {DATASET_FOLDER} use_context - {str(cmd_args.use_context)}")

    args = Arguments(train_csv=os.path.join(DATASET_FOLDER, "en.train_casimedicos.jsonl"),
                    test_csv=os.path.join(DATASET_FOLDER, "en.test_casimedicos.jsonl"),
                    dev_csv=os.path.join(DATASET_FOLDER, "en.dev_casimedicos.jsonl"),
                    #incorrect_ans = 0, # TODO, esto es de Paula, no del original
                    pretrained_model_name=PRETRAINED_MODEL,
                    use_context=cmd_args.use_context)
    
    #exp_name = f"4_5_ANS_(1.b)_ckpt_{model}@@@{os.path.basename(exp_dataset_folder)}@@@use_context{str(cmd_args.use_context)}@@@data{str(args.train_csv)}@@@seqlen{str(args.max_len)}".replace("/","_") # La linea de abajo es la adaptación hecha por mí de lo de Paula
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d-%H-%M")
    #exp_name = f"{model}@@@{os.path.basename(DATASET_FOLDER)}@@@use_context{str(cmd_args.use_context)}@@@data{str(args.train_csv)}@@@seqlen{str(args.max_len)}@@@execTime{str(formatted_datetime)}".replace("/","_")
    exp_name = f"{model}___data{os.path.basename(args.train_csv)}___seqlen{str(args.max_len)}___execTime{str(formatted_datetime)}".replace("/","_")

    train(gpu=args.gpu,
        args=args,
        #exp_dataset_folder=DATASET_FOLDER,
        experiment_name=exp_name,
        models_folder=MODELS_FOLDER,
        version=exp_name)
    
    time.sleep(60) # TODO, esto para qué sirve?