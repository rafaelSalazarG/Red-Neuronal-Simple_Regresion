# Red-Neuronal-Simple_Regresion
Este código implementa una red neuronal desde cero para resolver un problema de regresión no lineal en 3D.
1. Generación de datos (generar_datos_clasificacion)
Genera puntos distribuidos en una elipse tridimensional utilizando ecuaciones paramétricas.
Los datos tienen dos entradas (x, y) y una salida (z), que representa la variable objetivo.
2. Inicialización de pesos (inicializar_pesos)
Inicializa los pesos y sesgos de la red neuronal con valores aleatorios pequeños.
La red tiene:
Una capa de entrada con 2 neuronas (por las dos características de los datos).
Una capa oculta con un número configurable de neuronas.
Una capa de salida con 1 neurona (para la regresión).
3. Propagación hacia adelante (ejecutar_adelante)
Calcula las salidas de la red neuronal para un conjunto de entradas.
Utiliza:
ReLU como función de activación en la capa oculta.
Una función de activación lineal en la capa de salida.
4. Entrenamiento (train)
Entrena la red neuronal utilizando el algoritmo de backpropagation:
Propagación hacia adelante: Calcula las salidas de la red.
Cálculo de la pérdida:
Utiliza el error cuadrático medio (MSE) como función de pérdida.
Propagación hacia atrás:
Calcula los gradientes de la pérdida con respecto a los pesos y sesgos.
Ajusta los pesos y sesgos usando gradiente descendente.
Validación (opcional):
Evalúa la pérdida en un conjunto de validación y detiene el entrenamiento si la pérdida deja de mejorar.
5. Visualización
Durante el entrenamiento, grafica la pérdida en el conjunto de entrenamiento y, si corresponde, en el conjunto de validación.
También puede graficar los datos generados y la salida de la red neuronal en 3D.
6. Ejecución principal (iniciar)
Genera los datos de entrenamiento.
Inicializa los pesos de la red.
Entrena la red neuronal.
Grafica los resultados y devuelve los pesos entrenados.
