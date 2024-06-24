import evaluate
import json
import sklearn.metrics as metrics

'''
This script checks the metrics computed by Huggingface-evaluate and Sklearn libraries

This was done because the metrics computed by Huggingface-evaluate where quite strange

Script:
    1. Loads the confusion matrix from a json file
'''

# Cargar matriz de confusion
with open('chech_metrics_example.json', 'r') as file:
    eval_pred = json.load(file)

conf_matrix = eval_pred['eval_confusion_matrix']
print(conf_matrix)

etiquetas_reales = []
etiquetas_predichas = []

for index, lista_clase_real in enumerate(conf_matrix):
    for index_pred, cantidad in enumerate(lista_clase_real):
        etiquetas_reales += [index] * cantidad
        etiquetas_predichas += [index_pred] * cantidad

print(len(etiquetas_reales))
print(len(etiquetas_predichas))

def evaluate_hugginface(etiquetas_reales, etiquetas_predichas):
    f1 = evaluate.load('f1') # Load the f1 function
    precision = evaluate.load('precision') # Load the precision function
    recall = evaluate.load('recall') # Load the recall function

    predictions, labels = etiquetas_predichas, etiquetas_reales

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

    return {
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
    }

def metricas_a_mano(predictions, labels):
    f1_micro = metrics.f1_score(y_true=labels, y_pred=predictions, average='micro')
    f1_macro = metrics.f1_score(y_true=labels, y_pred=predictions, average='macro')
    f1_weighted = metrics.f1_score(y_true=labels, y_pred=predictions, average='weighted')
    f1_class = metrics.f1_score(y_true=labels, y_pred=predictions, average=None)

    precision_micro = metrics.precision_score(y_true=labels, y_pred=predictions, average='micro')
    precision_macro = metrics.precision_score(y_true=labels, y_pred=predictions, average='macro')
    precision_weighted = metrics.precision_score(y_true=labels, y_pred=predictions, average='weighted')
    precision_class = metrics.precision_score(y_true=labels, y_pred=predictions, average=None)

    recall_micro = metrics.recall_score(y_true=labels, y_pred=predictions, average='micro')
    recall_macro = metrics.recall_score(y_true=labels, y_pred=predictions, average='macro')
    recall_weighted = metrics.recall_score(y_true=labels, y_pred=predictions, average='weighted')
    recall_class = metrics.recall_score(y_true=labels, y_pred=predictions, average=None)

    return {
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'f1_opa': f1_class[0],
        'f1_opb': f1_class[1],
        'f1_opc': f1_class[2],
        'f1_opd': f1_class[3],

        'precision_micro': precision_micro,
        'precision_macro': precision_macro,
        'precision_weighted': precision_weighted,
        'precision_opa': precision_class[0],
        'precision_opb': precision_class[1],
        'precision_opc': precision_class[2],
        'precision_opd': precision_class[3],

        'recall_micro': recall_micro,
        'recall_macro': recall_macro,
        'recall_weighted': recall_weighted,
        'recall_opa': recall_class[0],
        'recall_opb': recall_class[1],
        'recall_opc': recall_class[2],
        'recall_opd': recall_class[3],
    }

metrics_hugging = evaluate_hugginface(etiquetas_reales, etiquetas_predichas)
metrics_sklearn = metricas_a_mano(etiquetas_reales, etiquetas_predichas)

print("Métricas \t| \tHuggingface \t| \tSklearn")
for key in metrics_hugging.keys():
    if metrics_hugging[key] == metrics_sklearn[key]:
        print(f"{key} \t| {metrics_hugging[key]} \t| {metrics_sklearn[key]} ✅")
    else:
        print(f"{key} \t| {metrics_hugging[key]} \t| {metrics_sklearn[key]} ❌")
      