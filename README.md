En este repositorio se incluye el código elaborado en el TFG "Modelos de supervivencia e inteligencia artificial explicable en la predicción del riesgo de conversión de deterioro cognitivo leve a Alzhéimer", realizado en la Universidad de Zaragoza en 2024.

En este trabajo se aporta una visión general completa de la eficacia de Random Survival Forests (RSF) y
Gradient Boosting Survival Analysis (GBSA) en el problema de la predicción de conversión de MCI a AD en un periodo de tiempo de hasta 5 años. Se han utilizado datos de pacientes con MCI del Alzheimer's Disease Neuroimaging Initiative (ADNI) en diferentes puntos temporales de su seguimiento (baseline, mes 12, mes 24, y una concatenación de baseline y mes 12). Asimismo, se ha elaborado un método simple pero efectivo para la obtención de métricas complementarias similares a las típicamente utilizadas en los modelos de aprendizaje automático tradicionales, y se ha utilizado [SHAP](https://github.com/shap/shap) para abordar la explicabilidad de los modelos y analizar los factores que más influyen sobre las predicciones realizadas.

Se incluyen dos programas en Python. A continuación se procede a explicar cómo utilizar cada uno de ellos y los requerimientos específicos. Además, también se incluyen en el directorio `data` varios conjuntos de datos de ejemplo.

### Imputación
La imputación de los datos se realiza mediante el programa `imputer.py`.

#### Requisitos
Se utiliza el paquete [missforest](https://github.com/epsilon-machine/missingpy), que requiere una versión no superior a la 1.1.3 de [scikit-learn](https://github.com/scikit-learn/scikit-learn). Se recomienda utilizar un entorno con la versión 3.8 de Python. Más detalles sobre las versiones de los paquetes utilizados en el fichero `imputer_requirements.txt`.

#### Uso
El programa `imputer.py` que se ejecuta siguiendo el siguiente formato:

`py -3.8 imputer.py -v <viscode> -k <n_folds>`

donde:
- `<viscode>` es el código de la visita que se quiere utilizar como referencia. Se puede usar cualquier visita, pero en este trabajo únicamente se han utilizado bl (por defecto), m12, m24 y bl+m12 (para datos longitudinales). Para que los datos se puedan leer correctamente, el nombre del fichero de entrada debe seguir el formato `ADNIMERGE_<viscode>.csv`.
- `<n_folds>` es el número de folds para la imputación utilizando k-fold. Por defecto se utilizan 5 folds.

Como resultado del programa, se generarán en el directorio `data/imputed/<viscode>` los ficheros necesarios con los datos imputados.

### Entrenamiento, evaluación y explicabilidad
La imputación de los datos se realiza mediante el programa `train_rsf.py` para Random Survival Forests (RSF) y el programa `train_gb.py` para Gradient Boosting Survival Analysis (GBSA).

#### Requisitos
Se utiliza el paquete [scikit-survival](https://github.com/sebp/scikit-survival), que requiere una versión igual o superior a la 1.3.2 de [scikit-learn](https://github.com/scikit-learn/scikit-learn). En este trabajo se ha utilizado un entorno con Python 3.11.8. Más detalles sobre las versiones de los paquetes utilizados en el fichero `train_requirements.txt`.

#### Uso
El programa `train.py` que se ejecuta siguiendo el siguiente formato:

`py -3.11 train.py -v <viscode> -k <n_folds>`

donde ambos parámetros son los mismos que en el programa de imputación.

La única diferencia entre ambos programas es la utilización de un modelo u otro. Este programa utiliza los ficheros generados por el programa de imputación, que deberán encontrarse en el directorio `data/imputed/<viscode>`.

El resultado de la ejecución del programa es el cálculo de las métricas de evaluación de c-index, accuracy, sensitivity, specificity y f1-score para cada fold, incluyendo media y desviación típica. Estas se muestran por la terminal en formato de tabla por terminal. También se obtiene una gráfica de explicabilidad global que agrupa los valores SHAP calculados para cada fold. Además, si se utilizan los datos de baseline, se calculan también las explicaciones locales de 4 pacientes concretos.
