import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np

file_path = "assets/dataset_clinicas.xlsx"  
df = pd.read_excel(file_path)

# Converter categorias em valores numéricos
credito_map = {"Bom": 2, "Regular": 1, "Ruim": 0}
df["Historico_Credito"] = df["Historico_Credito"].map(credito_map)

feedback_map = {"Interessado": 2, "Pouco Interessado": 1, "Não Interessado": 0}
df["Feedback_Clientes"] = df["Feedback_Clientes"].map(feedback_map)

# Definir features e target
X = df[["Faturamento_Mensal", "Num_Pacientes", "Historico_Credito", "Taxa_Conversao"]]
y = df["Feedback_Clientes"]  # Proxy para fechamento

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar e treinar o modelo
clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X_train, y_train)

# Fazer previsões
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Prever probabilidades para todas as clínicas
df["Probabilidade_Fechamento"] = np.round(clf.predict_proba(X)[:, 1], 2)

# Salvar resultados
output_path = "previsao_fechamento.xlsx"
df.to_excel(output_path, index=False)

print(f"Modelo treinado com precisão de {accuracy:.2f}")
print(f"Resultados salvos em {output_path}")
