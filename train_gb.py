import argparse
import warnings
warnings.simplefilter(action='ignore', category=DeprecationWarning)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.util import Surv


def delete_nan_values(ix_test):
    """
    Elimina valores vacíos en una lista de índices

    Parámetros:
    ix_test (list): Lista de índices de test de cada fold

    Devuelve:
    ix_test (list): Lista de índices sin valores vacíos
    """
    for i in range(len(ix_test)):
        for j in range(len(ix_test[i])):
            if np.isnan(ix_test[i][j]):
                ix_test[i].pop(j)
    return ix_test


def read_data(path, fold):
    """
    Carga los datos de un fold

    Parámetros:
    path (str): Ruta donde se encuentran los datos
    fold (int): Número de fold

    Devuelve:
    X_train (pd.DataFrame): Datos de entrenamiento
    y_train (np.array): Etiquetas de entrenamiento
    X_test (pd.DataFrame): Datos de test
    y_test (np.array): Etiquetas de test
    """
    X_train = pd.read_csv(f'{path}/X_train_{fold}.csv')
    y_train = Surv.from_dataframe('DX', 'M', pd.read_csv(f'{path}/y_train_{fold}.csv'))
    X_test = pd.read_csv(f'{path}/X_test_{fold}.csv')
    y_test = Surv.from_dataframe('DX', 'M', pd.read_csv(f'{path}/y_test_{fold}.csv'))
    return X_train, y_train, X_test, y_test


def predict_labels(surv, threshold=0.5):
    """
    Predice para cada paciente si progresará a Alzheimer o no en función de las probabilidades de
    supervivencia al final del periodo de seguimiento

    Parámetros:
    surv (np.array): Probabilidades de supervivencia de cada paciente
    threshold (float): Umbral que determina la pertenencia a una clase u otra (default: 0.5)

    """
    y_pred = np.zeros(len(surv))
    for i in range(len(surv)):
        if surv[i][-1] < threshold:
            y_pred[i] = 1
        else:
            y_pred[i] = 0
    return y_pred


def calculate_metrics(y_true, y_pred):
    """
    Calcula las métricas de evaluación de un clasificador binario

    Parámetros:
    y_true (np.array): Etiquetas reales
    y_pred (np.array): Etiquetas predichas

    Devuelve:
    accuracy (float): Tasa de acierto
    sensitivity (float): Sensibilidad
    specificity (float): Especificidad
    f1 (float): F1-score
    """
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return accuracy, sensitivity, specificity, f1


def evaluate_fold(gb, X_test, y_test, metrics):
    """
    Evalúa las predicciones con los datos de test de un fold

    Parámetros:
    gb (sksurv.ensemble.GradientBoostingSurvivalAnalysis): Modelo GBSA entrenado
    X_test (pd.DataFrame): Datos de test
    y_test (np.array): Etiquetas de test
    metrics (dict): Diccionario de métricas de evaluación para todos los folds

    Devuelve:
    metrics (dict): Diccionario actualizado con las métricas del fold actual
    """
    score = gb.score(X_test, y_test)
    surv = gb.predict_survival_function(X_test, return_array=True)
    y_pred = predict_labels(surv)
    y_true = [y[0].astype(int) for y in y_test]
    accuracy, sensitivity, specificity, f1 = calculate_metrics(y_true, y_pred)
    metrics['c-index'].append(score)
    metrics['acc'].append(accuracy)
    metrics['sens'].append(sensitivity)
    metrics['spec'].append(specificity)
    metrics['f1'].append(f1)
    return metrics


def metrics_summary(metrics):
    """
    Muestra un resumen con las métricas de evaluación de todos los folds, incluyendo la media y la
    desviación típica

    Parámetros:
    metrics (dict): Diccionario de métricas de evaluación para todos los folds

    Devuelve:
    metrics_df (pd.DataFrame): DataFrame con las métricas de evaluación de todos los folds
    """
    metrics_df = pd.DataFrame(metrics)
    metrics_df['fold'] = range(1, len(metrics_df) + 1)
    metrics_df.set_index('fold', inplace=True)
    print(metrics_df)
    print(metrics_df.describe().loc[['mean', 'std']])
    return metrics_df


def explain_local(gb, explainer, X_test, y_test):
    """
    Explica las predicciones de un conjunto de pacientes seleccionados

    Parámetros:
    gb (sksurv.ensemble.GradientBoostingSurvivalAnalysis): Modelo GBSA entrenado
    explainer (shap.Explainer): Explicador de SHAP
    X_test (pd.DataFrame): Datos de test
    y_test (np.array): Etiquetas de test
    """
    surv = gb.predict_survival_function(X_test, return_array=True)
    y_pred = np.array(predict_labels(surv))
    y_true = np.array([y[0].astype(int) for y in y_test])
    patients = [2, 13, 10, 0]
    X_test_sel = X_test.iloc[patients]
    y_test_sel = y_test[patients]
    shap_values = explainer(X_test_sel)
    for i in range(len(patients)):
        ix = patients[i]
        print(f'Patient {i + 1}: y_true={y_true[ix]}, y_pred={y_pred[ix]}, months={y_test[ix][1]}, survival={surv[ix]}')
        plt.figure()
        shap.plots.waterfall(shap_values[i], show=False)
        plt.tight_layout()
        plt.show()
    plot_survival_function(gb, X_test_sel, y_test_sel)


def plot_survival_function(gb, X_test, y_test):
    """
    Dibuja las funciones de supervivencia de un conjunto de pacientes seleccionados

    Parámetros:
    gb (sksurv.ensemble.GradientBoostingSurvivalAnalysis): Modelo GBSA entrenado
    X_test (pd.DataFrame): Datos de test
    y_test (np.array): Etiquetas de test
    """
    surv = gb.predict_survival_function(X_test, return_array=True)
    plt.figure()
    for i, s in enumerate(surv[:2]):
        label = f'sMCI#{i + 1}'
        plt.plot(gb.unique_times_, s, label=label)
    for i, s in enumerate(surv[2:], start=2):
        label = f'pMCI#{i - 1}'
        line, = plt.plot(gb.unique_times_, s, label=label)
        plt.axvline(x=y_test[i][1], linestyle='--', color=line.get_color())
    plt.ylabel('Survival probability')
    plt.xlabel('Time in months')
    plt.xticks(np.arange(gb.unique_times_.min(), gb.unique_times_.max() + 1, 12))
    plt.legend()
    plt.grid(True, alpha=0.5, linestyle='--')
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', dest='viscode', type=str, default='bl',
                        help='Código de visita utilizada como referencia (default: bl)',
                        choices=['bl', 'm12', 'm24', 'bl+m12'])
    parser.add_argument('-k', dest='n_folds', type=int, default=5,
                        help='Número de folds (default: 5)')
    args = parser.parse_args()

    X = pd.read_csv(f'data/X_{args.viscode}.csv')
    ix_test = pd.read_csv(f'data/ix_test_{args.viscode}.csv', header=None).values.tolist()
    ix_test = delete_nan_values(ix_test)
    new_index = [ix for ix_test_fold in ix_test for ix in ix_test_fold]

    metrics = {
        'c-index': [],
        'acc': [],
        'sens': [],
        'spec': [],
        'f1': []
    }
    shap_values_per_fold = []
    gb = GradientBoostingSurvivalAnalysis(n_estimators=1000, learning_rate=0.01, random_state=1234)
    for fold in range(1, args.n_folds + 1):
        X_train, y_train, X_test, y_test = read_data(f'data/imputed/{args.viscode}', fold)
        gb.fit(X_train, y_train)
        metrics = evaluate_fold(gb, X_test, y_test, metrics)
        explainer = shap.Explainer(gb.predict, X_train, feature_names=X_train.columns, seed=1234)
        shap_values = explainer.shap_values(X_test)
        for shaps in shap_values:
            shap_values_per_fold.append(shaps)

    metrics_summary(metrics)
    if args.viscode == 'bl':
        explain_local(rsf, explainer, X.reindex(new_index), y_test)
    plt.figure()
    shap.summary_plot(np.array(shap_values_per_fold), X.reindex(new_index))


if __name__ == '__main__':
    main()
