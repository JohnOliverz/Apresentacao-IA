import pandas as pd
import numpy as np
import re
import string

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_wine

# 📥 Carregar o conjunto de dados Wine
wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df["target"] = wine.target  # Adiciona a coluna de rótulos (0, 1 ou 2)

# 🔄 Normalizando os dados para melhor desempenho do SVM
scaler = StandardScaler()
X = scaler.fit_transform(df.drop("target", axis=1))  # Normaliza os atributos
y = df["target"]

# 📊 Divisão dos dados em treino e teste (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🏆 Criando e treinando o modelo SVM
svm_model = SVC(kernel='linear', C=1.0)
svm_model.fit(X_train, y_train)

# 🔍 Fazer previsões no conjunto de teste
y_pred_svm = svm_model.predict(X_test)

# 📊 Avaliação do modelo
print("📢 Resultados do SVM no dataset Wine:")
print(f"Acurácia: {accuracy_score(y_test, y_pred_svm):.2f}")
print(classification_report(y_test, y_pred_svm))
