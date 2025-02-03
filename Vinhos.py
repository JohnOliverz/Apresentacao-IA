import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA  # Importando PCA
from mlxtend.plotting import plot_decision_regions

# 📥 Carregar o conjunto de dados Wine
wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df["target"] = wine.target  # Adiciona a coluna de rótulos (0, 1 ou 2)

# 📊 Visualizar a distribuição das classes
plt.figure(figsize=(6, 4))
sns.countplot(x=df["target"], palette="Set2")
plt.xlabel("Classe do Vinho")
plt.ylabel("Contagem")
plt.title("Distribuição das Classes do Conjunto de Dados Wine")
plt.show()

# 🔄 Normalizando os dados
scaler = StandardScaler()
X = scaler.fit_transform(df.drop("target", axis=1))  # Normaliza os atributos
y = df["target"]

# 📊 Divisão dos dados em treino e teste (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 🏆 Criando e treinando o modelo SVM com kernel RBF
svm_model_rbf = SVC(kernel='rbf', C=1.0, gamma='scale')  # Usando kernel RBF
svm_model_rbf.fit(X_train, y_train)

# 🔍 Fazer previsões no conjunto de teste
y_pred_svm = svm_model_rbf.predict(X_test)

# 📊 Avaliação do modelo
accuracy = accuracy_score(y_test, y_pred_svm)
print("📢 Resultados do SVM com Kernel RBF no dataset Wine:")
print(f"✅ Precisão do modelo: {accuracy:.2f}\n")

# Criando um dicionário para mapear os rótulos numéricos para nomes compreensíveis
wine_classes = {
    0: "Classe 0 - Tipo A",
    1: "Classe 1 - Tipo B",
    2: "Classe 2 - Tipo C"
}

# 📊 Exibir relatório de classificação com nomes de classes personalizados
print("📊 Relatório de classificação:")
print(classification_report(y_test, y_pred_svm, target_names=[wine_classes[i] for i in range(3)]))

# 📌 Gerando a matriz de confusão
cm = confusion_matrix(y_test, y_pred_svm)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine_classes.values(), yticklabels=wine_classes.values())
plt.xlabel("Classe Predita")
plt.ylabel("Classe Real")
plt.title("Matriz de Confusão - SVM com Kernel RBF")
plt.show()

# 📌 Aplicando PCA para reduzir para 2D (apenas para visualização)
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# 🏆 Treinando o modelo novamente para visualização com PCA
svm_vis = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_vis.fit(X_train_pca, y_train)

# 📊 Gráfico de separação das classes usando as duas componentes principais
plt.figure(figsize=(8, 6))
plot_decision_regions(X_train_pca, y_train.to_numpy(), clf=svm_vis, legend=2)  # Conversão de y_train para array NumPy
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title("Separação das Classes de Vinho (PCA + Kernel RBF)")
plt.show()
