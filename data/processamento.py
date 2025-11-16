"""
ETAPA 3 — MODELO BASELINE PARA YOUTUBE_VIEWS
============================================

Este modelo baseline irá prever o número de visualizações
(views) de vídeos do YouTube.

Objetivo:
- Criar modelo simples de regressão linear
- Obter métricas base
- Gerar os gráficos obrigatórios
- Salvar o modelo
"""

# ============================================================================
# IMPORTS
# ============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# PASSO 1 — CARREGAR DADOS
# ============================================================================
print("="*70)
print("ETAPA 3 — MODELO BASELINE PARA YOUTUBE_VIEWS")
print("="*70)

df = pd.read_csv("youtube_views.csv")
print(f"✓ Dados carregados: {df.shape[0]} linhas, {df.shape[1]} colunas\n")
print(df.head())

# ============================================================================
# PASSO 2 — DEFINIR TARGET E FEATURES
# ============================================================================

# TODO: AJUSTAR após você me enviar as colunas reais
TARGET_COLUMN = "views"          # <- SEU ALVO REAL
COLUNAS_REMOVER = ["video_id"]   # <- AJUSTAR DEPOIS

X = df.drop(columns=[TARGET_COLUMN] + COLUNAS_REMOVER, errors="ignore")
y = df[TARGET_COLUMN]

print(f"\nFeatures usadas ({len(X.columns)} colunas):")
print(X.columns.tolist())

# ============================================================================
# PASSO 3 — DIVIDIR TREINO / VALIDAÇÃO / TESTE
# ============================================================================
RANDOM_STATE = 42

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=RANDOM_STATE
)

print("\nDivisão dos dados:")
print(f"Treino:     {len(X_train)}")
print(f"Validação:  {len(X_val)}")
print(f"Teste:      {len(X_test)}")

# ============================================================================
# PASSO 4 — TREINAR MODELO
# ============================================================================
print("\nTreinando modelo baseline...")
modelo = LinearRegression()
modelo.fit(X_train, y_train)
print("✓ Modelo treinado!")

# ============================================================================
# PASSO 5 — PREVISÕES
# ============================================================================
y_train_pred = modelo.predict(X_train)
y_val_pred = modelo.predict(X_val)

print("✓ Previsões geradas")

# ============================================================================
# PASSO 6 — MÉTRICAS
# ============================================================================
def metricas(y_true, y_pred, nome):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n{nome}")
    print("-"*40)
    print(f"MSE:  {mse:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE:  {mae:.3f}")
    print(f"R²:   {r2:.3f}")

    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}

print("\nMÉTRICAS DO MODELO BASELINE")
metricas_treino = metricas(y_train, y_train_pred, "Treino")
metricas_val    = metricas(y_val,    y_val_pred,    "Validação")

# ============================================================================
# PASSO 7 — GRÁFICO: PREVISTO VS REAL
# ============================================================================
plt.figure(figsize=(10,6))
plt.scatter(y_val, y_val_pred, alpha=0.6, edgecolors='k')
minv = min(y_val.min(), y_val_pred.min())
maxv = max(y_val.max(), y_val_pred.max())
plt.plot([minv, maxv], [minv, maxv], 'r--')
plt.xlabel("Real")
plt.ylabel("Previsto")
plt.title("Predições vs Reais (Validação)")
plt.tight_layout()
plt.savefig("pred_vs_real.png", dpi=300)
print("✓ Gráfico salvo: pred_vs_real.png")
plt.show()

# ============================================================================
# PASSO 8 — DISTRIBUIÇÃO DOS RESÍDUOS
# ============================================================================
res = y_val - y_val_pred
plt.figure(figsize=(10,6))
plt.hist(res, bins=30, edgecolor='black', alpha=0.7)
plt.axvline(0, color='red', linestyle='--')
plt.xlabel("Resíduos")
plt.title("Distribuição dos Resíduos")
plt.tight_layout()
plt.savefig("residuos.png", dpi=300)
print("✓ Gráfico salvo: residuos.png")
plt.show()

# ============================================================================
# PASSO 9 — IMPORTÂNCIA DAS FEATURES (coeficientes)
# ============================================================================
coef_df = pd.DataFrame({
    "feature": X_train.columns,
    "coef": modelo.coef_
}).sort_values("coef", key=abs, ascending=False)

print("\nCoeficientes do modelo:")
print(coef_df)

plt.figure(figsize=(10,7))
plt.barh(coef_df["feature"].head(10), coef_df["coef"].head(10))
plt.title("Top 10 Features mais importantes")
plt.tight_layout()
plt.savefig("importancia.png", dpi=300)
print("✓ Gráfico salvo: importancia.png")
plt.show()

# ============================================================================
# PASSO 10 — SALVAR MODELO
# ============================================================================
joblib.dump(modelo, "baseline_youtube.pkl")
print("\n✓ Modelo salvo: baseline_youtube.pkl")
