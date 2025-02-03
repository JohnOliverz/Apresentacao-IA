import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA  # Importando PCA

# 📌 Carregar o dataset
df = pd.read_csv(r'C:\Users\Jhonatan\OneDrive\Área de Trabalho\IA - Data Mining\Hotel Reservations.csv')

# 📌 Verificar valores ausentes antes de criar a coluna de classificação
df = df.dropna(subset=['avg_price_per_room'])

# 📌 Criar a nova coluna 'label_avg_price_per_room'
df['label_avg_price_per_room'] = pd.cut(
    df['avg_price_per_room'], 
    bins=[-float('inf'), 85, 115, float('inf')],  
    labels=[1, 2, 3]
)

# 📌 Remover valores NaN gerados e converter para inteiro
df = df.dropna(subset=['label_avg_price_per_room'])
df['label_avg_price_per_room'] = df['label_avg_price_per_room'].astype(int)

# 📌 Remover colunas irrelevantes
df = df.drop(columns=['avg_price_per_room', 'Booking_ID', 'arrival_year', 'arrival_month', 'arrival_date'])

# 📌 Converter variáveis categóricas para numéricas
df = pd.get_dummies(df)

# 📌 Separar variáveis independentes (X) e dependentes (y)
X = df.drop(columns=['label_avg_price_per_room'])
y = df['label_avg_price_per_room']

# 📌 Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 📌 Aplicar SMOTE para balancear os dados
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# 📌 Normalizar os dados para SVM
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 📌 Treinar o modelo SVM com kernel RBF e ajuste de hiperparâmetros
svm_model_rbf = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
svm_model_rbf.fit(X_train, y_train)

# 📌 Fazer previsões no conjunto de teste
y_pred_rbf = svm_model_rbf.predict(X_test)

# 📌 Avaliar o modelo e exibir os resultados
accuracy_rbf = accuracy_score(y_test, y_pred_rbf)
print("📢 Resultados do SVM com Kernel RBF no dataset de Hotéis:")
print(f"✅ Precisão do modelo: {accuracy_rbf:.2f}\n")
print("📊 Relatório de classificação:")
print(classification_report(y_test, y_pred_rbf, target_names=["Preço Baixo (1)", "Preço Médio (2)", "Preço Alto (3)"]))

# 📊 Exibir a matriz de confusão
plt.figure(figsize=(8, 6))
conf_matrix_rbf = confusion_matrix(y_test, y_pred_rbf)
sns.heatmap(conf_matrix_rbf, annot=True, fmt='d', cmap='Blues', xticklabels=["Preço Baixo (1)", "Preço Médio (2)", "Preço Alto (3)"], yticklabels=["Preço Baixo (1)", "Preço Médio (2)", "Preço Alto (3)"])
plt.title("Matriz de Confusão - SVM com Kernel RBF")
plt.xlabel("Predição")
plt.ylabel("Real")
plt.show()

# 📊 Visualizar a distribuição de classes após o balanceamento
plt.figure(figsize=(6, 4))
sns.countplot(x=y_train, palette="Set2")
plt.xlabel("Classe de Preço (Treinamento)")
plt.ylabel("Contagem")
plt.title("Distribuição das Classes no Conjunto de Treinamento (Após SMOTE)")
plt.show()

# 📌 Aplicando PCA para reduzir para 2D (apenas para visualização)
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)

# 📊 Gráfico de separação das classes usando as duas primeiras componentes principais
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_train_pca[:, 0], y=X_train_pca[:, 1], hue=y_train, palette="Set2", s=100, edgecolor='black', marker='o')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title("Separação das Classes de Preço (PCA com Kernel RBF)")
plt.show()

# 📌 Exportar o dataset alterado
df.to_csv('Hotel_Reservations_Processed.csv', index=False)
