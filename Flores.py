import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# 📥 Carregar o conjunto de dados IRIS
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df = pd.read_csv(url)

# 🔍 Verificando as primeiras linhas do dataset
print(df.head())

# 🔄 Convertendo a coluna "species" para valores numéricos
label_encoder = LabelEncoder()
df['species'] = label_encoder.fit_transform(df['species'])

# 🔠 Separação das variáveis de entrada (X) e saída (y)
X = df.drop(columns=['species'])  # Remove a coluna de rótulo
y = df['species']  # Apenas os rótulos

# 📊 Divisão dos dados em treino e teste (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🏆 Criando e treinando o modelo SVM
svm_model = SVC(kernel='linear', C=1.0)
svm_model.fit(X_train, y_train)

# 🔍 Fazer previsões no conjunto de teste
y_pred_svm = svm_model.predict(X_test)

# 📊 Avaliação do modelo
print("📢 Resultados do SVM:")
print(f"Acurácia: {accuracy_score(y_test, y_pred_svm):.2f}")
print(classification_report(y_test, y_pred_svm, target_names=label_encoder.classes_))
