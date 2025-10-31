import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Análise de Machine Learning - Completa", layout="wide")

# Função de carregamento
@st.cache_resource
def load_models_and_data():
    kmeans = joblib.load("models/kmeans_model.pkl")
    scaler_unsup = joblib.load("models/scaler.pkl")
    df_unsup = pd.read_csv("data/df_clusters.csv")

    lasso = joblib.load("models/lasso_model.pkl")
    scaler_sup = joblib.load("models/scaler_supervisionado.pkl")
    df_sup = pd.read_csv("data/df_supervisionado.csv")

    return kmeans, scaler_unsup, df_unsup, lasso, scaler_sup, df_sup


# Carregamento dos arquivos
try:
    kmeans, scaler_unsup, df_unsup, lasso, scaler_sup, df_sup = load_models_and_data()
    st.success("✅ Modelos e dados carregados com sucesso!")
except Exception as e:
    st.error("Erro ao carregar arquivos. Verifique se os diretórios e nomes estão corretos.")
    st.stop()

# Menu lateral
st.sidebar.title("Navegação")
pagina = st.sidebar.radio(
    "Escolha a seção:",
    ["Visão Geral", "Aprendizado Não Supervisionado", "Aprendizado Supervisionado"]
)


# Página 1 – Visão Geral
if pagina == "Visão Geral":
    st.title("📊 Análise Completa de Machine Learning")
    st.markdown("""
    Este aplicativo combina **Aprendizado Não Supervisionado (K-Means)** e **Supervisionado (Lasso Regression)**.  
    Ele permite explorar clusters identificados e realizar previsões baseadas em novas entradas.
    """)
    st.image("https://cdn-icons-png.flaticon.com/512/2779/2779775.png", width=120)


# Página 2 – NÃO SUPERVISIONADO
elif pagina == "Aprendizado Não Supervisionado":
    st.title("🤖 Análise Não Supervisionada - K-Means")

    st.subheader("Visualização dos Clusters")
    st.dataframe(df_unsup.head())

    st.markdown("#### Mapa de Calor das Médias por Cluster")
    cluster_means = df_unsup.groupby("cluster").mean()

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(cluster_means, annot=True, cmap="viridis", fmt=".2f", ax=ax)
    st.pyplot(fig)

    st.markdown("#### Distribuição de Amostras por Cluster")
    fig2, ax2 = plt.subplots()
    df_unsup["cluster"].value_counts().plot(kind="bar", color="teal", ax=ax2)
    plt.title("Número de Amostras por Cluster")
    st.pyplot(fig2)

    st.subheader("Prever Cluster de Novo Registro")

    col1, col2, col3, col4 = st.columns(4)
    age = col1.number_input("Idade (padronizada)", value=0.0)
    bmi = col2.number_input("IMC (padronizado)", value=0.0)
    bp = col3.number_input("Pressão (padronizada)", value=0.0)
    s5 = col4.number_input("S5 (padronizado)", value=0.0)

    if st.button("🔮 Prever Cluster"):
        new_data = pd.DataFrame([[age, bmi, bp, s5]], columns=["age", "bmi", "bp", "s5"])

        # Adiciona colunas faltantes com valor 0 (mesmo formato do treino)
        for col in ["sex", "s1", "s2", "s3", "s4", "s6"]:
            new_data[col] = 0.0

        new_data = new_data[["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]]

        new_scaled = scaler_unsup.transform(new_data)
        cluster_pred = kmeans.predict(new_scaled)[0]

        st.success(f"O novo registro pertence ao **Cluster {cluster_pred}**")


# Página 3 – SUPERVISIONADO
elif pagina == "Aprendizado Supervisionado":
    st.title("🧮 Predição - Regressão Supervisionada (Lasso)")

    st.markdown("""
    O modelo supervisionado usa **Lasso Regression** para prever o valor da variável alvo (`target`)
    com base em novas entradas padronizadas.
    """)

    st.subheader("Inserir Dados de Entrada")

    col1, col2, col3, col4 = st.columns(4)
    age = col1.number_input("Idade (padronizada)", value=0.0)
    sex = col2.number_input("Sexo (padronizado)", value=0.0)
    bmi = col3.number_input("IMC (padronizado)", value=0.0)
    bp = col4.number_input("Pressão (padronizada)", value=0.0)

    col5, col6, col7, col8, col9, col10 = st.columns(6)
    s1 = col5.number_input("S1", value=0.0)
    s2 = col6.number_input("S2", value=0.0)
    s3 = col7.number_input("S3", value=0.0)
    s4 = col8.number_input("S4", value=0.0)
    s5 = col9.number_input("S5", value=0.0)
    s6 = col10.number_input("S6", value=0.0)

    if st.button("Prever Target"):
        new_data = pd.DataFrame([[age, sex, bmi, bp, s1, s2, s3, s4, s5, s6]],
                                columns=["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"])

        new_scaled = scaler_sup.transform(new_data)
        prediction = lasso.predict(new_scaled)[0]

        st.success(f"Predição estimada da variável alvo: **{prediction:.2f}**")

    st.markdown("### Correlação entre as Variáveis (Supervisionado)")
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    sns.heatmap(df_sup.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax3)
    st.pyplot(fig3)
