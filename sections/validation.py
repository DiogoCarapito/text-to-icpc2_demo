import streamlit as st
from utils.utils import load_csv_github, load_predictions_labels
import pandas as pd
import numpy as np
import plotly.express as px


def validation():
    st.title("Validação")
    st.write(
        "Validação para já **apenas** na classificação correta da **descrição ICPC-2** de cada um 726 códigos e não com a lista completa de doenças. Espera-se que classificações com mais volume de dados tenham mais previsões corretas. O **split treino/teste aleatório** pode ter influência na capacidade de classificação do modelo. Primeiro quero melhorar performace do modelo nas descrições base antes de avançar para as inclusões de todas as doenças."
    )

    st.divider()

    # load the frequency table
    df_frequency_model_result = load_csv_github(
        "https://raw.githubusercontent.com/DiogoCarapito/text-to-icpc2/main/data/data_pre_train.csv"
    )

    # count the frequency of each code
    frequency_table = df_frequency_model_result["code"].value_counts().reset_index()

    # merge the frequency table with the original dataset to get the chapter and origin
    frequency_table = df_frequency_model_result.merge(
        frequency_table, on="code", how="left"
    )

    # order by the code (index keeps the code order)
    frequency_table = frequency_table.sort_index()

    # add a log2 count to the frequency_table for better visualization
    frequency_table["count_log2"] = frequency_table["count"].apply(lambda x: np.log2(x))

    # load the correct predictions with a runid
    # runid is defined in the sidebar and is used to load the correct predictions
    # if
    df_correct_current_model = load_predictions_labels(st.session_state["runid"])

    # merge frequency_table with correct_prediction into df_frequency_model_result
    df_frequency_model_result = frequency_table.merge(
        df_correct_current_model, on="code", how="left"
    )

    # Assuming df_correct_current_model has a column indicating correctness, e.g., 'is_correct'
    df_frequency_model_result["is_correct"] = df_frequency_model_result[
        "is_correct"
    ].fillna(False)

    # create a version of the dataset with only the icpc2_description origin (only 726 codes)
    df_frequency_model_result_icpc2 = df_frequency_model_result[
        df_frequency_model_result["origin"] == "icpc2_description"
    ]

    # filter section for exploration
    # col_1_val, col_3_val, col_4_val = st.columns(4, vertical_alignment="bottom")
    col_1_val, col_2_val, col_3_val = st.columns(3, vertical_alignment="bottom")

    with col_1_val:
        # filter by code
        filter_code_val = st.text_input(
            "Filtrar por código",
            help="Filtrar por código ICPC específico (ex: T90)",
            key="filter_code_val",
        )

    with col_2_val:
        # filter by text/description
        filter_text_val = st.text_input(
            "Filtrar por texto/descrição",
            help="Filtrar por texto/descrição específico (ex: diabetes mellitus)",
            key="filter_text_val",
        )

    with col_3_val:
        # filter by chapter
        filter_chapter_val = st.multiselect(
            "Filtrar por capítulo",
            df_frequency_model_result_icpc2["chapter"].unique(),
            default=None,
            help="Filtrar por capítulo específico (ex: K)",
            key="filter_chapter_val",
        )

    # df_frequency_model_result_filter = df_frequency_model_result.copy()
    df_frequency_model_result_icpc2_filter = df_frequency_model_result_icpc2.copy()

    # logic to filter the dataset
    if filter_code_val:
        df_frequency_model_result_icpc2_filter = df_frequency_model_result_icpc2_filter[
            df_frequency_model_result_icpc2_filter["code"].str.contains(
                filter_code_val, case=False
            )
        ]

    if filter_text_val:
        df_frequency_model_result_icpc2_filter = df_frequency_model_result_icpc2_filter[
            df_frequency_model_result_icpc2_filter["text"].str.contains(
                filter_text_val, case=False
            )
        ]

    if filter_chapter_val:
        df_frequency_model_result_icpc2_filter = df_frequency_model_result_icpc2_filter[
            df_frequency_model_result_icpc2_filter["chapter"].isin(filter_chapter_val)
        ]

    # metric with the number of correct predictions and percentage
    col_1_1, col_1_2, col_1_3 = st.columns(3)
    num_correct = df_frequency_model_result_icpc2_filter[
        df_frequency_model_result_icpc2_filter["is_correct"] == True
    ].shape[0]

    # num of codes
    num_codes = df_frequency_model_result_icpc2_filter.shape[0]
    percentage_correct = round(100 * num_correct / num_codes, 2)

    with col_1_1:
        st.metric(
            "Total de registros selecionados",
            df_frequency_model_result_icpc2_filter.shape[0],
        )
    with col_1_2:
        st.metric("Número de previsões corretas", num_correct)
    with col_1_3:
        st.metric("Percentagem de previsões corretas", f"{percentage_correct}%")

    log2_toggle = st.checkbox(
        "Contagens em Log2",
        key="count_log2_toggle",
        help="Mostrar as contagens em escala logarítmica para facilitar a visualização",
    )

    # Create a bar chart with Plotly
    fig = px.bar(
        df_frequency_model_result_icpc2_filter,
        x="code",
        y="count_log2" if log2_toggle else "count",
        color="is_correct",
        color_discrete_map={True: "green", False: "red"},
        title="Previsões Corretas ",
        # hover_data={"code": True, "text": True, "chapter": True},
    )

    # Customize hover data
    fig.update_traces(
        hovertemplate="<b>Código:</b> %{x}<br>"
        + "<b>Texto:</b> %{customdata[1]}<extra></extra><br>"
        + "<b>Contagem:</b> %{y}<br>"
        + "<b>Capítulo:</b> %{customdata[0]}<br>",
        customdata=df_frequency_model_result_icpc2_filter[["chapter", "text"]],
    )

    # Sort the x-axis alphabetically
    fig.update_layout(xaxis={"categoryorder": "category ascending"})

    # Display the Plotly chart in Streamlit
    st.plotly_chart(fig)
