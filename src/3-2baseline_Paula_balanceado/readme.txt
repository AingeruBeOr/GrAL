Esto es todo adaptado del repo de Paula: https://github.com/paulaonta/medmcqa

Este experimento, a diferencia del anterior, busca balancear el número de opciones correctas 
en el dataset para que el modelo no se aprenda la distribución de clases.

Cuidado con el batch size!! Una instancia realmente pasan a ser 25 inputs para el modelo. 

Instancia:
    - Instancia con respuesta correcta en A.
        - Input con la opción A: <s>Q</s>opa</s>
        - Input con la opción B: <s>Q</s>opb</s>
        - ...(+3)
    - Instancia con respuesta correcta en B.
        - ...(+5)
    - Instancia con respuesta correcta en C.
        - ...(+5)
    - Instancia con respuesta correcta en D.
        - ...(+5)
    - Instancia con respuesta correcta en E.
        - ...(+5)

# Sweeps with wandb

1. Generar un .yaml
2. Crear un sweep: wandb sweep --project <project_name> <YAML file>. Este nos devuleve el 'sweep_ID'
3. Start an agent: wandb agent <entity>/<project_name>/<sweep_ID>

El sweep está mal y está en el proyecto 'tfg-baseline-casimedicos-synthetic' de WANDB.

# Upload local runs to WANDB

Example: wandb sync -p tfg-baseline-casimedicos-synthetic *