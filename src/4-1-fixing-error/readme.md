Como en el experimento "4-MedMCQA_and_CasiMedicos" no mejoraba el val_acc, he lanzado un entrenamiento de prueba para comprobar si el error es el script o los datos. He copiado "4-MedMCQA_and_CasiMedicos" y he creado el dataset nuevo que acabo de mencionar.

En este experimento, el dataset de entrenamiento consiste en CasiMedicos-balanced y una instancia (balanceada, es decir, la opción correcta en las 4 posiciones posibles) de MedMCQA.

Conculusiones:
- Primer epoch constante el val_acc.
- A partir del 3-4-5 empieza a subir; es decir, los primeros epochs "desaprende"

Cambios (comparando los archivos de "4-1-fixing-error" y "4-1-fixing-error-v_old"):
- `dataset.py`: no cambia
- `model.py`: mis cambios (sobre todo el de calcular el confusion matrix (puede que el accuracy para cuando el batch_size sea diferente esté mal))
- `train.py`: mis cambios (cambios de sintaxis, cambios de valores hardcodeados y el parámetro val_check_interval)
- `args.py`: mis cambios (pero manteniendo los parametros comunes igual que en "4-1-fixing-error-v_old" y el parámetro val_check_interval)

Conclusiones:
- los cambios en train.py no causan el problema de que no mejore
- los cambios en model.py no cuasan el problema de que no mejore
- puede que sea :
    - `val_check_interval`, el resto de hiperparámetros o `model.py`
    - el dataset
- Si pongo `batch_size=1` y `gradient_acc=2` no da lo mismo que `batch_size=2` y `gradient_acc=1` (comparar los entrenamientos `2024-05-08-19-37` y `2024-05-09-10-14`):
    - puede ser porque no es determinista (hay un parámetro del trainer que se puede configurar). Si se hace en dos GPUs no es determinista de por sí. Hay que poner en el trainer `Deterministic = true`. Al no se determinista, las GPUs ignoran ciertas cosas que lo hacen medio-random y eso hace que no se vuelva a poder reproducir.
    - puede ser que no esté bien configurado el cálculo del loss. Iker dice que calcula el del batch_size inmediato y no el del real.
        - entiendo que se calcula el immediato pero no se actualiza hasta que han pasado `gradient_acc` número de steps (https://github.com/Lightning-AI/pytorch-lightning/blob/e0307277a03c0822c26b525c1cdfa71425ed0214/docs/source-fabric/advanced/gradient_accumulation.rst#L4) (https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html)
        - **Conclusión final**: El Cross Entropy loss tiene que hacer la suma de los batches, no la media como está por defecto porque entonces batch_size=1 y gradient_acc=2 no es lo mismo que batch_size=2 y gradient_acc=1. Entonces, ponemos eso como "sum" en vez de como "mean" que está puesto por defecto (es el valor por defecto). (si puedo, intento explicarlo en el TFG porque es un cambio importante en comparación con lo que tenían hecho los de MedMCQA)