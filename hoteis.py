import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Carregar o dataset
df = pd.read_csv(r'C:\Users\Jhonatan\OneDrive\Área de Trabalho\IA - Data Mining\Hotel Reservations.csv', delimiter=',')


# 🔹 Verificar valores ausentes antes de criar a coluna de classificação
if df['avg_price_per_room'].isnull().sum() > 0:
    print("⚠️ Há valores ausentes na coluna 'avg_price_per_room'. Linhas com NaN serão removidas.")
    df = df.dropna(subset=['avg_price_per_room'])

# 🔹 Criar a nova coluna 'label_avg_price_per_room'
df['label_avg_price_per_room'] = pd.cut(
    df['avg_price_per_room'], 
    bins=[-float('inf'), 85, 115, float('inf')],  # Permite valores negativos, caso existam
    labels=[1, 2, 3]
)

# 🔹 Remover possíveis valores NaN gerados na nova coluna
df = df.dropna(subset=['label_avg_price_per_room'])

# 🔹 Converter a coluna label_avg_price_per_room para inteiro (necessário para a classificação)
df['label_avg_price_per_room'] = df['label_avg_price_per_room'].astype(int)

# 🔹 Remover colunas desnecessárias
df = df.drop(columns=['avg_price_per_room', 'Booking_ID', 'arrival_year', 'arrival_month', 'arrival_date'])

# 🔹 Converter colunas categóricas para numéricas usando one-hot encoding
df = pd.get_dummies(df)

# 🔹 Separar os dados em variáveis de entrada (X) e saída (y)
X = df.drop(columns=['label_avg_price_per_room'])
y = df['label_avg_price_per_room']

# 🔹 Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 🔹 Treinar o modelo SVM
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# 🔹 Fazer previsões no conjunto de teste
y_pred = svm_model.predict(X_test)

# 🔹 Avaliar o modelo e exibir os resultados
accuracy = accuracy_score(y_test, y_pred)
print("📢 Resultados do SVM no dataset de Hotéis:")
print(f"Acurácia: {accuracy:.2f}")
print(classification_report(y_test, y_pred))

# 🔹 Exportar o dataset alterado
df.to_csv('Hotel_Reservations_Processed.csv', index=False)