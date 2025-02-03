import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import load_breast_cancer
from mlxtend.plotting import plot_decision_regions

# 📥 Carregar o conjunto de dados Breast Cancer
cancer = load_breast_cancer()
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df["target"] = cancer.target  # Adiciona a coluna de rótulos (0 ou 1)

# 📊 Visualizar a distribuição das classes
plt.figure(figsize=(6, 4))
sns.countplot(x=df["target"], palette="Set2")
plt.xlabel("Classe do Tumor")
plt.ylabel("Contagem")
plt.title("Distribuição das Classes do Conjunto de Dados Breast Cancer")
plt.show()

# 🔄 Normalizando os dados
scaler = StandardScaler()
X = scaler.fit_transform(df.drop("target", axis=1))  # Normaliza os atributos
y = df["target"]

# 📊 Divisão dos dados em treino e teste (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 🏆 Criando e treinando o modelo SVM com kernel linear
svm_model_linear = SVC(kernel='linear', C=1.0)  # Usando kernel linear
svm_model_linear.fit(X_train, y_train)

# 🔍 Fazer previsões no conjunto de teste
y_pred_svm = svm_model_linear.predict(X_test)

# 📊 Avaliação do modelo
accuracy = accuracy_score(y_test, y_pred_svm)
print("📢 Resultados do SVM com Kernel Linear no dataset Breast Cancer:")
print(f"✅ Precisão do modelo: {accuracy:.2f}\n")

# 📊 Relatório de classificação
cancer_classes = {
    0: "Maligno",
    1: "Benigno"
}
print("📊 Relatório de classificação:")
print(classification_report(y_test, y_pred_svm, target_names=[cancer_classes[i] for i in range(2)]))

# 📌 Gerando a matriz de confusão
cm = confusion_matrix(y_test, y_pred_svm)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=cancer_classes.values(), yticklabels=cancer_classes.values())
plt.xlabel("Classe Predita")
plt.ylabel("Classe Real")
plt.title("Matriz de Confusão - SVM com Kernel Linear")
plt.show()

# 📊 Visualizando a separação das classes (para 2 características apenas)
# Vamos usar apenas duas características para a visualização da fronteira de decisão
X_train_vis = X_train[:, :2]  # Usando apenas as duas primeiras características
X_test_vis = X_test[:, :2]

# 🏆 Treinando o modelo novamente para visualização com as duas características
svm_vis = SVC(kernel='linear', C=1.0)
svm_vis.fit(X_train_vis, y_train)

# 📊 Gráfico de separação das classes usando as duas primeiras características
plt.figure(figsize=(8, 6))
plot_decision_regions(X_train_vis, y_train.to_numpy(), clf=svm_vis, legend=2)  # Conversão de y_train para array NumPy
plt.xlabel(df.columns[0])
plt.ylabel(df.columns[1])
plt.title("Separação das Classes de Tumor (Kernel Linear - 2 Características)")
plt.show()
