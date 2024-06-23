import argparse
import os
import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sklearn.neighbors._base
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base

import pandas as pd
from missingpy import MissForest
from sklearn.model_selection import KFold


def export_data(X_train, X_test, y_train, y_test, path, i):
    """
    Exporta los datos imputados en un fold en varios ficheros CSV

    Parámetros:
    X_train (pd.DataFrame): Datos de entrenamiento imputados
    X_test (pd.DataFrame): Datos de test imputados
    y_train (pd.DataFrame): Etiquetas de entrenamiento
    y_test (pd.DataFrame): Etiquetas de test
    path (str): Ruta donde se guardarán los ficheros
    i (int): Número de fold
    """
    if not os.path.exists(path):
        os.makedirs(path)
    X_train.to_csv(f'{path}/X_train_{i}.csv', index=False)
    X_test.to_csv(f'{path}/X_test_{i}.csv', index=False)
    y_train.to_csv(f'{path}/y_train_{i}.csv', index=False)
    y_test.to_csv(f'{path}/y_test_{i}.csv', index=False)


def impute_fold(imputer, X_train, X_test):
    """
    Imputa los datos de un fold

    Parámetros:
    imputer (missingpy.MissForest): Imputador de datos
    X_train (pd.DataFrame): Datos de entrenamiento
    X_test (pd.DataFrame): Datos de test

    Devuelve:
    X_train_imputed (pd.DataFrame): Datos de entrenamiento imputados
    X_test_imputed (pd.DataFrame): Datos de test imputados
    """
    print('Train')
    X_train_imputed = imputer.fit_transform(X_train)
    print('Test')
    X_test_imputed = imputer.fit_transform(X_test)
    X_train_imputed = pd.DataFrame(X_train_imputed, columns=X_train.columns)
    X_test_imputed = pd.DataFrame(X_test_imputed, columns=X_test.columns)
    return X_train_imputed, X_test_imputed


def impute_data(X, y, n_folds, path):
    """
    Imputa los valores faltantes de un conjunto de datos en varios folds y los exporta en
    ficheros CSV

    Parámetros:
    X (pd.DataFrame): Conjunto de datos
    y (pd.DataFrame): Etiquetas
    n_folds (int): Número de folds
    path (str): Ruta donde se guardarán los ficheros

    Devuelve:
    ix_test (list): Índices de test de cada fold
    """
    if not os.path.exists(path):
        os.makedirs(path)
    ix_training = []
    ix_test = []
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=1234)
    for fold in kf.split(X):
        ix_training.append(fold[0])
        ix_test.append(fold[1])
    for fold, (train_outer_ix, test_outer_ix) in enumerate(zip(ix_training, ix_test), start=1):
        print(f'Imputing fold {fold}')
        X_train, X_test = X.iloc[train_outer_ix, :], X.iloc[test_outer_ix, :]
        y_train, y_test = y.iloc[train_outer_ix], y.iloc[test_outer_ix]
        imputer = MissForest(random_state=1234)
        X_train_imputed, X_test_imputed = impute_fold(imputer, X_train, X_test)
        export_data(X_train_imputed, X_test_imputed, y_train, y_test, path, fold)
    return ix_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', dest='viscode', type=str, default='bl',
                        help='Código de visita utilizada como referencia (default: bl)',
                        choices=['bl', 'm12', 'm24', 'bl+m12'])
    parser.add_argument('-k', dest='n_folds', type=int, default=5,
                        help='Número de folds (default: 5)')
    args = parser.parse_args()

    X = pd.read_csv(f'data/ADNIMERGE_{args.viscode}.csv')
    X['DX'] = X['DX'].replace({'MCI': False, 'Dementia': True})
    if 'PTGENDER' in X.columns:
        X['PTGENDER'] = X['PTGENDER'].replace({'Male': 0, 'Female': 1})
    y = X[['DX', 'M']]
    X.drop(columns=['RID', 'DX', 'M'], axis=1, inplace=True)
    ix_test = impute_data(X, y, args.n_folds, f'data/imputed/{args.viscode}')
    pd.DataFrame(X).to_csv(f'data/X_{args.viscode}.csv', index=False)
    pd.DataFrame(ix_test).to_csv(f'data/ix_test_{args.viscode}.csv', index=False, header=False)


if __name__ == '__main__':
    main()
