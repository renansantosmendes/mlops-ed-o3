import os
import pickle
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

"""# 2 - Fazendo a leitura do dataset e atribuindo às respectivas variáveis"""
data = pd.read_csv('https://raw.githubusercontent.com/'
                   'renansantosmendes/lectures-cdas-2023'
                   '/master/fetal_health.csv')

"""# 3 - Preparando o dado antes de iniciar o treino do modelo"""
features_to_remove = data.columns[7:]
X = data.drop(features_to_remove, axis=1)
y = data["fetal_health"]
columns = list(X.columns)
scaler = preprocessing.StandardScaler()
X_df = scaler.fit_transform(X)
X_df = pd.DataFrame(X_df, columns=columns)
X_train, X_test, y_train, y_test = train_test_split(X_df,
                                                    y,
                                                    random_state=42,
                                                    test_size=0.3)

os.environ['MLFLOW_TRACKING_USERNAME'] = 'renansantosmendes'
os.environ['MLFLOW_TRACKING_PASSWORD'] = '2f9986ad8ae5fcc143' \
                                         '16c61038e3781805eef9dd'
mlflow.set_tracking_uri('https://dagshub.com/renansantosmendes/'
                        'mlops-puc-210823.mlflow')
mlflow.sklearn.autolog()

"""# **Modelos Ensemble**"""
gradient_clf = GradientBoostingClassifier(max_depth=10, n_estimators=150, learning_rate=0.01)
with mlflow.start_run(run_name='gradiente_bosting') as run:
    gradient_clf.fit(X_train, y_train)
