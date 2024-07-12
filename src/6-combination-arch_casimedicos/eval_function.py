import numpy as np
import evaluate
import wandb

'''
Custom evaluation function to compute metrics on the evaluation dataset during training (trainer.train()), 
evaluate (trainer.evaluate()) and inference (trainer.test()) methods.

Computes the metrics from huggingface evaluate library.

It also cretes a confusion matrix to plot into wandb.
'''

class_names = ['a', 'b', 'c', 'd', 'e']

# Set evaluetaion function to be used during training (accuracy, f1, ...)
accuracy = evaluate.load('accuracy')  # Load the accuracy function
f1 = evaluate.load('f1')  # Load the f-score function
precision = evaluate.load('precision')  # Load the precision function
recall = evaluate.load('recall')  # Load the recall function
cnf_matrix = evaluate.load('BucketHeadP65/confusion_matrix')  # Load the confusion matrix function

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy_value = accuracy.compute(predictions=predictions, references=labels)

    f1_value_micro = f1.compute(predictions=predictions, references=labels, average='micro')
    f1_value_macro = f1.compute(predictions=predictions, references=labels, average='macro')
    f1_value_weighted = f1.compute(predictions=predictions, references=labels, average='weighted')
    class_f1 = f1.compute(predictions=predictions, references=labels, average=None)

    precision_value_micro = precision.compute(predictions=predictions, references=labels, average='micro')
    precision_value_macro = precision.compute(predictions=predictions, references=labels, average='macro')
    precision_value_weighted = precision.compute(predictions=predictions, references=labels, average='weighted')
    class_precision = precision.compute(predictions=predictions, references=labels, average=None, zero_division='warn')

    recall_value_micro = recall.compute(predictions=predictions, references=labels, average='micro')
    recall_value_macro = recall.compute(predictions=predictions, references=labels, average='macro')
    recall_value_weighted = recall.compute(predictions=predictions, references=labels, average='weighted')
    class_recall = recall.compute(predictions=predictions, references=labels, average=None, zero_division='warn')

    confusion_matrix_serializable = cnf_matrix.compute(predictions=predictions, references=labels)

    #  Every element in the return dict, must be serializable so we log confusion_matrix (which is a plot object from wanbd library) independently to wandb
    confusion_matrix = wandb.plot.confusion_matrix(probs=None, y_true=labels, preds=predictions, class_names=class_names, title='Train eval confusion matrix')
    wandb.log({'confusion_matrix': confusion_matrix})

    return {
        'accuracy': accuracy_value['accuracy'],
        'f1_micro': f1_value_micro['f1'],
        'f1_macro': f1_value_macro['f1'],
        'f1_weighted': f1_value_weighted['f1'],
        'f1_opa': class_f1['f1'][0],
        'f1_opb': class_f1['f1'][1],
        'f1_opc': class_f1['f1'][2],
        'f1_opd': class_f1['f1'][3],
        'precision_micro': precision_value_micro['precision'],
        'precision_macro': precision_value_macro['precision'],
        'precision_weighted': precision_value_weighted['precision'],
        'precision_opa': class_precision['precision'][0],
        'precision_opb': class_precision['precision'][1],
        'precision_opc': class_precision['precision'][2],
        'precision_opd': class_precision['precision'][3],
        'recall_micro': recall_value_micro['recall'],
        'recall_macro': recall_value_macro['recall'],
        'recall_weighted': recall_value_weighted['recall'],
        'recall_opa': class_recall['recall'][0],
        'recall_opb': class_recall['recall'][1],
        'recall_opc': class_recall['recall'][2],
        'recall_opd': class_recall['recall'][3],
        'confusion_matrix': confusion_matrix_serializable['confusion_matrix'].tolist()  # convert to list to be serializable
    }