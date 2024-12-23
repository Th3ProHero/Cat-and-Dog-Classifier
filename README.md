# Cat-and-Dog-Classifier
Este proyecto contiene varias implementaciones de redes neuronales para clasificar imágenes de gatos y perros utilizando PyTorch. Los modelos fueron entrenados usando un dataset de imágenes de gatos y perros.
# Clasificador de Imágenes: Gatos vs Perros

Este proyecto implementa un clasificador de imágenes basado en una red neuronal convolucional (CNN) para diferenciar entre dos clases de imágenes: gatos y perros. Utilizando la biblioteca PyTorch, el modelo se entrena con un conjunto de datos de imágenes, que se cargan desde dos directorios: uno para entrenamiento y otro para validación. 

## Funcionamiento del Código

El proceso comienza con la carga y preprocesamiento de las imágenes. Las imágenes se redimensionan a 128x128 píxeles, se convierten en tensores y se normalizan. Luego, el modelo de red neuronal, denominado `CatDogClassifier`, consta de dos bloques principales: un conjunto de capas convolucionales para extraer características relevantes de las imágenes, seguido de capas completamente conectadas que generan la predicción final entre las dos clases: gatos y perros.

El entrenamiento se realiza durante 10 épocas, utilizando el optimizador Adam y la función de pérdida de entropía cruzada. Una vez entrenado, el modelo es evaluado en un conjunto de validación y se calcula su precisión. Finalmente, el modelo entrenado se guarda en un archivo para su posterior uso.

Este clasificador puede cargarse nuevamente para hacer predicciones sobre nuevas imágenes.

## Descripción de la Versión 1

Esta primera versión del clasificador de imágenes utiliza una arquitectura más simple en comparación con la versión optimizada. Sin embargo, incluye los componentes básicos necesarios para realizar la clasificación de imágenes de gatos y perros, con algunos aspectos clave:

1. **Preprocesamiento Básico de Imágenes:**  
   El modelo realiza un preprocesamiento de las imágenes mediante transformaciones básicas que incluyen el cambio de tamaño a 128x128 píxeles, la conversión de las imágenes a tensores y la normalización de los valores de píxeles. Aunque estas transformaciones son fundamentales, no incluyen técnicas de aumento de datos, lo que puede limitar la capacidad del modelo para generalizar.

2. **Arquitectura Simple:**  
   El modelo tiene una arquitectura de red neuronal convolucional (CNN) básica con dos capas de convolución seguidas de capas de pooling. Esta estructura es adecuada para tareas simples de clasificación de imágenes, pero es menos profunda que la versión optimizada y no captura características tan complejas de las imágenes.

3. **Entrenamiento con Pérdida de Entropía Cruzada:**  
   Se usa la función de pérdida `CrossEntropyLoss` y el optimizador `Adam` para entrenar el modelo. La configuración es adecuada para tareas de clasificación binaria, como la clasificación de gatos y perros.

4. **Entrenamiento por 10 Épocas:**  
   El modelo se entrena durante 10 épocas, lo que es un número adecuado para tareas simples, aunque en modelos más complejos o para mejorar la precisión, se podría aumentar este valor.

5. **Validación y Precisión:**  
   El modelo se evalúa en un conjunto de validación para calcular la precisión. Esto permite conocer el rendimiento del modelo en datos no vistos durante el entrenamiento.

6. **Guardar y Cargar el Modelo:**  
   El modelo entrenado se guarda en un archivo `.pth` para poder reutilizarlo posteriormente sin necesidad de entrenarlo de nuevo. Este archivo se puede cargar en cualquier momento para hacer predicciones.

## Mejoras en la version 2:

Este código introduce varias mejoras clave sobre el modelo original:

1. **Aumento de Datos (Data Augmentation):**  
   Se ha incorporado el uso de técnicas de aumento de datos para mejorar la capacidad del modelo de generalizar. Las imágenes se someten a transformaciones como rotaciones aleatorias y voltear horizontalmente, lo que permite crear variaciones de las imágenes de entrenamiento y reducir el riesgo de sobreajuste.

2. **Modelo Optimizado con Más Capas:**  
   El modelo se ha ampliado con una arquitectura más profunda, añadiendo una capa adicional de convolución (con 128 filtros) para extraer características más complejas de las imágenes. Esto ayuda a mejorar el rendimiento en tareas más complicadas, como la clasificación de imágenes.

3. **Normalización de Lotes (Batch Normalization):**  
   Se ha añadido normalización de lotes después de cada capa convolucional para mejorar la estabilidad del entrenamiento y acelerar la convergencia. Esta técnica ayuda a evitar problemas de gradientes explosivos o desvanecidos.

4. **Regularización con Dropout:**  
   Se ha implementado la técnica de Dropout en las capas completamente conectadas para evitar el sobreajuste. Esto ayuda a mejorar la capacidad de generalización del modelo, especialmente en redes profundas.

5. **Ajuste Dinámico del Learning Rate (Scheduler):**  
   Se ha incorporado un scheduler de tasa de aprendizaje que ajusta dinámicamente el learning rate durante el entrenamiento. Esto permite un entrenamiento más eficiente y puede ayudar a mejorar la convergencia al reducir la tasa de aprendizaje de forma gradual.

6. **Mayor Número de Épocas:**  
   El número de épocas se ha incrementado a 20 para permitir un entrenamiento más largo y exhaustivo, con la esperanza de que el modelo pueda mejorar su precisión a lo largo del tiempo.

## Requisitos

- Python 3.x
- PyTorch
- PIL
- Tkinter
- Matplotlib
- CUDA
- NVIDIA GPU

## Hardware utilizado para este proyecto
-CPU: i7 10700F
-GPU: RTX 4070
-RAM: 32 GB DDR4
SSD NVME de preferencia.

## Uso

1. Preprocesa y carga tus imágenes en el formato adecuado (una carpeta para imágenes de gatos y otra para perros).
2. Estructura de carpetas:
  train/
    perros/
    gatos/
  val/
    perros/
    gatos/
3. El dataset utilizado se puede descargar de: https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset tambien puedes usar https://www.kaggle.com/datasets/chetankv/dogs-cats-images
4. Ejecuta el script para entrenar el modelo.
5. Evalúa el modelo con las imágenes de validación.
6. Utiliza el modelo entrenado para hacer predicciones sobre nuevas imágenes.
