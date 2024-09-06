import streamlit as st

# import timeit
from sections.demo import demo
from sections.dataset import dataset
from sections.validation import validation

# from utils.utils import (
#     device_cuda_mps_cpu,
#     load_model,
#     prediction_display,
#     load_csv_github,
# )
# import pandas as pd
# import numpy as np
# import plotly.express as px


def main():
    st.set_page_config(
        page_title="text-to-icpc2 Demo",
        page_icon=":ledger:",
        #layout="wide",
        initial_sidebar_state="auto",
    )

    demo()

    st.divider()

    dataset()

    st.divider()

    validation()


# st.title("Explorador do Dataset de treino")

# st.write(
#     "Este é um explorador do dataset de treino, onde é possível filtrar os registros por código, texto, capítulo e origem."
# )
# st.write(
#     "O dataset de treino está disponivel em [https://huggingface.co/diogocarapito/text-to-icpc2](https://huggingface.co/diogocarapito/text-to-icpc2)"
# )

# # load the pos-processed dataset
# dataset_train = load_csv_github(
#     "https://raw.githubusercontent.com/DiogoCarapito/text-to-icpc2/main/data/data_pre_train.csv"
# )

# # filter section for exploration
# st.write("## Filtros para exploração")
# col_1, col_2, col_3, col_4 = st.columns(4, vertical_alignment="bottom")
# with col_1:
#     # filter by code
#     filter_code = st.text_input(
#         "Filtrar por código",
#         help="Filtrar por código ICPC específico (ex: T90)",
#     )

# with col_2:
#     # filter by text/description
#     filter_text = st.text_input(
#         "Filtrar por texto/descrição",
#         help="Filtrar por texto/descrição específico (ex: diabetes mellitus)",
#     )

# with col_3:
#     # filter by chapter
#     filter_chapter = st.multiselect(
#         "Filtrar por capítulo",
#         dataset_train["chapter"].unique(),
#         default=None,
#         help="Filtrar por capítulo específico (ex: K)",
#     )

# with col_4:
#     # filter by origin
#     filter_origin = st.multiselect(
#         "Filtrar por origem",
#         dataset_train.origin.unique(),
#         default=None,
#         help="Filtrar por origem específica dos dados(ex: icpc2_description)",
#     )

# # logic to filter the dataset
# if filter_code:
#     dataset_train = dataset_train[
#         dataset_train["code"].str.contains(filter_code, case=False)
#     ]

# if filter_text:
#     dataset_train = dataset_train[
#         dataset_train["text"].str.contains(filter_text, case=False)
#     ]
# if filter_chapter:
#     dataset_train = dataset_train[dataset_train["chapter"].isin(filter_chapter)]

# if filter_origin:
#     dataset_train = dataset_train[dataset_train["origin"].isin(filter_origin)]

# st.metric("Total de registros selecionados", dataset_train.shape[0])

# # show filtered dataset
# st.dataframe(dataset_train, use_container_width=True, hide_index=True)

# st.divider()

# # data sources
# st.subheader("Fontes")
# data_sources = (
#     "https://github.com/DiogoCarapito/text-to-icpc2/blob/main/data/data_sources.csv"
# )
# st.write(data_sources)


if __name__ == "__main__":
    main()
