import streamlit as st

# import timeit
from sections.demo import demo
from sections.dataset import dataset
from sections.validation import validation
from sections.sidebar_info import sidebar_info

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
        # layout="wide",
        initial_sidebar_state="auto",
    )

    # info sobre o projeto em side bar
    sidebar_info()

    # create 3 tabs
    tab_demo, tab_dataset, tab_validacao = st.tabs(["Demo", "Dataset", "Validação"])

    with tab_demo:
        # interface da demo de interação com o modelo e render da predição com dados de contexto
        demo()

    with tab_dataset:
        # exploração do dataset
        dataset()

    with tab_validacao:
        # exploração da validação do dataset
        validation()


if __name__ == "__main__":
    main()
