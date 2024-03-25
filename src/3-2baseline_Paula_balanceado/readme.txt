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