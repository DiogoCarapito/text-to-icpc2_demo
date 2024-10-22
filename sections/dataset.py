import streamlit as st
from utils.utils import load_csv_github
import pandas as pd
import plotly.express as px


def dataset():
    st.title("Explorador do Dataset de treino")

    st.write(
        "Este é um explorador do dataset de treino, onde é possível filtrar os registros por código, texto, capítulo e origem."
    )
    st.write(
        "O dataset de treino está disponivel em [https://huggingface.co/datasets/diogocarapito/text-to-icpc2](https://huggingface.co/datasets/diogocarapito/text-to-icpc2)"
    )

    # load the pos-processed dataset
    dataset_train = load_csv_github(
        "https://raw.githubusercontent.com/DiogoCarapito/text-to-icpc2/main/data/data_pre_train.csv"
    )

    # filter section for exploration
    col_1, col_2, col_3, col_4 = st.columns(4, vertical_alignment="bottom")
    with col_1:
        # filter by code
        filter_code = st.text_input(
            "Filtrar por código",
            help="Filtrar por código ICPC específico (ex: T90)",
        )

    with col_2:
        # filter by text/description
        filter_text = st.text_input(
            "Filtrar por texto/descrição",
            help="Filtrar por texto/descrição específico (ex: diabetes mellitus)",
        )

    with col_3:
        # filter by chapter
        filter_chapter = st.multiselect(
            "Filtrar por capítulo",
            dataset_train["chapter"].unique(),
            default=None,
            help="Filtrar por capítulo específico (ex: K)",
        )

    with col_4:
        # filter by origin
        filter_origin = st.multiselect(
            "Filtrar por origem",
            dataset_train.origin.unique(),
            default=None,
            help="Filtrar por origem específica dos dados(ex: icpc2_description)",
        )

    # logic to filter the dataset
    if filter_code:
        dataset_train = dataset_train[
            dataset_train["code"].str.contains(filter_code, case=False)
        ]

    if filter_text:
        dataset_train = dataset_train[
            dataset_train["text"].str.contains(filter_text, case=False)
        ]
    if filter_chapter:
        dataset_train = dataset_train[dataset_train["chapter"].isin(filter_chapter)]

    if filter_origin:
        dataset_train = dataset_train[dataset_train["origin"].isin(filter_origin)]

    st.metric("Total de registros selecionados", dataset_train.shape[0])

    # count the frequency of each code
    frequency_table = dataset_train["code"].value_counts().reset_index()

    # Remove duplicates from dataset_train based on 'code'
    dataset_train_unique = dataset_train.drop_duplicates(subset="code")

    frequency_table = dataset_train_unique.merge(frequency_table, on="code", how="left")

    # Create a stacked bar chart with Plotly
    fig = px.bar(
        frequency_table,
        x="code",
        y="count",
        color="origin",
        color_discrete_map={
            "gpt-4o-mini_human-dc": "red",
            "human-dc": "red",
            "gpt-4o-mini": "green",
            "icpc2_description": "blue",
            "icpc2_short": "blue",
            "icpc2_inclusion": "blue",
            "icd10_description": "blue",
        },
        title="Distribuição dos códigos ICPC2 no dataset de treino",
        hover_data={"code": True, "text": True, "chapter": True},
    )

    # Sort the x-axis alphabetically
    fig.update_layout(xaxis={"categoryorder": "category ascending"}, barmode="stack")

    tab_tabela, tab_grafico = st.tabs(["Tabela", "Gráfico"])

    with tab_tabela:
        # show filtered dataset
        st.dataframe(dataset_train, use_container_width=True, hide_index=True)

    with tab_grafico:
        # Display the Plotly chart in Streamlit
        st.plotly_chart(fig)
