import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mlxtend.plotting import plot_decision_regions

# 📥 Carregar o conjunto de dados IRIS
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df = pd.read_csv(url)

# 📊 Exibir a distribuição das classes no dataset IRIS
plt.figure(figsize=(6, 4))
sns.countplot(x='species', data=df, palette="Set2")
plt.xlabel("Classe da Espécie")
plt.ylabel("Contagem")
plt.title("Distribuição das Classes no Conjunto de Dados IRIS")
plt.show()

# 🔄 Convertendo a coluna "species" para valores numéricos
label_encoder = LabelEncoder()
df['species'] = label_encoder.fit_transform(df['species'])

# 🔠 Separação das variáveis de entrada (X) e saída (y)
X = df.drop(columns=['species'])  # Remove a coluna de rótulo
y = df['species']  # Apenas os rótulos

# 📊 Divisão dos dados em treino e teste (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 🔄 Normalizando os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 🔍 Aplicar PCA (redução de dimensionalidade para 2 componentes principais)
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# 🏆 Criando e treinando o modelo SVM com Kernel RBF
svm_model_rbf = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_model_rbf.fit(X_train_pca, y_train)

# 🔍 Fazer previsões no conjunto de teste
y_pred_rbf = svm_model_rbf.predict(X_test_pca)

# 📊 Avaliação do modelo
accuracy_rbf = accuracy_score(y_test, y_pred_rbf)
print("📢 Resultados do SVM com Kernel RBF e PCA no dataset IRIS:")
print(f"✅ Precisão do modelo: {accuracy_rbf:.2f}\n")

# Criando um dicionário para mapear os rótulos numéricos para nomes compreensíveis
iris_classes = {i: label_encoder.classes_[i] for i in range(len(label_encoder.classes_))}

# 📊 Exibir relatório de classificação com nomes de classes personalizados
print("📊 Relatório de classificação:")
print(classification_report(y_test, y_pred_rbf, target_names=[iris_classes[i] for i in range(3)]))

# 📊 Plotar a matriz de confusão
cm = confusion_matrix(y_test, y_pred_rbf)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris_classes.values(), yticklabels=iris_classes.values())
plt.xlabel("Classe Predita")
plt.ylabel("Classe Real")
plt.title("Matriz de Confusão")
plt.show()

# 🎨 Visualizar a fronteira de decisão com PCA (apenas 2D após PCA)
plt.figure(figsize=(8, 6))
plot_decision_regions(X_train_pca, y_train.to_numpy(), clf=svm_model_rbf, legend=2)
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title("Separação das Classes pelo SVM com Kernel RBF (após PCA)")
plt.show()
