import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# ===============================
# CONFIGURAÇÃO BÁSICA
# ===============================
st.set_page_config(page_title="Análise de Clusters", layout="wide")
st.title("Análise de Clusters - Modelo KMeans")

# ===============================
# CARREGAMENTO DOS ARQUIVOS
# ===============================
@st.cache_resource
def load_model_and_data():
    kmeans = joblib.load("kmeans_model.pkl")
    scaler = joblib.load("scaler.pkl")
    df = pd.read_csv("df_clusters.csv")
    return kmeans, scaler, df

kmeans, scaler, df = load_model_and_data()

st.success("Modelo e dados carregados com sucesso!")

# ===============================
# EXIBIÇÃO DOS DADOS
# ===============================
st.subheader("Dados com Clusters")
st.dataframe(df.head())

# ===============================
# MÉDIAS POR CLUSTER
# ===============================
st.subheader("Média das Variáveis por Cluster")
cluster_means = df.groupby("cluster").mean(numeric_only=True)
st.dataframe(cluster_means)

# ===============================
# HEATMAP DE CORRELAÇÕES ENTRE VARIÁVEIS
# ===============================
st.subheader("Heatmap - Média das variáveis por cluster")

# Seleciona apenas as variáveis numéricas originais (sem target nem cluster)
variaveis = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]
cluster_means = df.groupby("cluster")[variaveis].mean()

plt.figure(figsize=(10, 6))
sns.heatmap(cluster_means.T, cmap="coolwarm", annot=True, fmt=".3f")
plt.title("Média das variáveis por cluster")
st.pyplot(plt)


# ===============================
# PREDIÇÃO PARA NOVOS DADOS
# ===============================
st.subheader("Teste de Novo Registro")

col1, col2, col3, col4 = st.columns(4)
age = col1.number_input("Idade (padronizada)", value=0.0)
bmi = col2.number_input("IMC (padronizado)", value=0.0)
bp = col3.number_input("Pressão (padronizada)", value=0.0)
s5 = col4.number_input("S5 (padronizado)", value=0.0)

if st.button("🔮 Prever Cluster"):
    new_data = pd.DataFrame([[age, bmi, bp, s5]], columns=["age", "bmi", "bp", "s5"])
    
    # Adiciona as colunas faltantes com valor 0
    for col in ["sex", "s1", "s2", "s3", "s4", "s6"]:
        new_data[col] = 0.0

    # Reordena as colunas na mesma ordem usada no treino
    new_data = new_data[["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]]

    # Transforma e prediz
    new_scaled = scaler.transform(new_data)
    cluster_pred = kmeans.predict(new_scaled)[0]
    
    st.success(f"O novo registro pertence ao **Cluster {cluster_pred}**")


# ===============================
# VISUALIZAÇÃO FINAL (Clusters)
# ===============================
st.subheader("Distribuição dos Clusters")
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="cluster", palette="Set2")
plt.title("Distribuição dos Clusters")
st.pyplot(plt)