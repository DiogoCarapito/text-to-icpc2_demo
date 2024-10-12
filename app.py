import streamlit as st
from sections.sidebar_info import sidebar_info
from sections.demo import demo
from sections.dataset import dataset

# from sections.validation import validation
# from sections.split import split


def main():
    st.set_page_config(
        page_title="text-to-icpc2 Demo",
        page_icon=":ledger:",
        # layout="wide",
        initial_sidebar_state="auto",
    )

    #st.session_state["runid"] = "862e53bb1e7a4c05ab8a049c5a97a257"

    # info sobre o projeto em side bar
    sidebar_info()

    # create 3 tabs
    # tab_demo, tab_dataset, tab_validacao, tab_split = st.tabs(
    #     ["Demo", "Dataset", "Validação", "Split"]
    # )

    tab_demo, tab_dataset = st.tabs(["Demo", "Dataset"])

    with tab_demo:
        # interface da demo de interação com o modelo e render da predição com dados de contexto
        demo()

    with tab_dataset:
        # exploração do dataset
        dataset()

    # with tab_validacao:
    #     # exploração da validação do dataset
    #     validation()

    # with tab_split:
    #     # exploração do split treino/teste
    #     split()


if __name__ == "__main__":
    main()
