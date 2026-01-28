# Bibliotecas necessárias
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.utils import resample


#-------------------------------------------------------------------------------
# Leitura dos dados
ev = pd.read_csv("evasao_toyexample.csv", sep=';', encoding='ANSI')
ev = ev[ev['forma_evasao'] != 'Sem evasão']

# Correção e criação de colunas
ev['evasao'] = ev['evasao'].astype('category')
ev['ira'] = ev['ira'].str.replace(',', '.').astype(float)
ev['proprep'] = ev['proprep'].str.replace(',', '.').astype(float)
ev['Area_the'] = ev['Area_the'].astype('category')
ev['turno'] = ev['turno'].astype('category')
ev['grau'] = pd.Categorical(ev['grau'], categories=["Bacharelado", "ABI", "Tecnológico", "Licenciatura"], ordered=False)
ev['setor'] = ev['setor'].astype('category')

#-------------------------------------------------------------------------------
# Balanceamento dos dados
evadidos = ev[ev['evasao'] == 1]
n_evadidos = ev[ev['evasao'] == 0]
n_evadidos_sorteados = resample(n_evadidos, replace=False, n_samples=len(evadidos), random_state=123)
db = pd.concat([evadidos, n_evadidos_sorteados])

#-------------------------------------------------------------------------------
# Seleção de variáveis e codificação
X = db[['ira', 'setor', 'proprep', 'grau', 'turno']]
y = db['evasao'].astype(int)

# Codificação de variáveis categóricas
X_encoded = pd.get_dummies(X, drop_first=True)

#-------------------------------------------------------------------------------
# Divisão em treino e teste (70% treino, 30% teste)
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.3, stratify=y, random_state=123
)

#-------------------------------------------------------------------------------
# Ajuste do modelo de Regressão Logística
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Coeficientes estimados
coeficientes = pd.Series(model.coef_[0], index=X_train.columns)
print("Coeficientes estimados:\n", coeficientes)

# Intercepto (termo constante)
print("\nIntercepto:", model.intercept_[0])
#-------------------------------------------------------------------------------
# Predições
y_pred_prob = model.predict_proba(X_test)[:, 1]
y_pred_class = (y_pred_prob > 0.5).astype(int)

#-------------------------------------------------------------------------------
# Avaliação do modelo
conf_matrix = confusion_matrix(y_test, y_pred_class)
print("Matriz de Confusão:\n", conf_matrix)

print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred_class))




#-------------------------------------------------------------------------------
### Comparação de modelos ###
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

# Lista de modelos
modelos = {
    'Regressão Logística': LogisticRegression(max_iter=1000),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'SVM RBF': SVC(kernel='rbf', C=1, gamma=0.1),
    'Árvore de Decisão': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(n_estimators=500),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1),
    'Rede Neural': MLPClassifier(hidden_layer_sizes=(5,), alpha=0.1, max_iter=1000)
}

#-------------------------------------------------------------------------------
# Avaliação dos modelos
for nome, modelo in modelos.items():
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', modelo)
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    print(f"\n🔍 Modelo: {nome}")
    print("Matriz de Confusão:")
    print(confusion_matrix(y_test, y_pred))
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred))
