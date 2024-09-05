import streamlit as st

# from transformers import pipeline
# from datasets import load_dataset
# import numpy as np
# import torch
import timeit

from utils.utils import (
    device_cuda_mps_cpu,
    load_model,
    # load_val_dataset,
    # add_labels_to_prediction,
    prediction_display,
    # load_labels_dataframe,
    load_csv_github,
)


def main():
    st.set_page_config(
        page_title="text-to-icpc2 Demo",
        page_icon=":ledger:",
        layout="wide",
        initial_sidebar_state="auto",
    )

    st.title("text-to-icpc2 Demo")
    expander_descrição = st.expander("Descrição do projeto")
    with expander_descrição:
        st.write(
            "Este é um demo do modelo text-to-icpc2, onde é possível classificar um diagnóstico em um código ICPC-2"
        )
        st.write(
            "O modelo foi treinado com dados de diagnósticos de saúde e códigos ICPC-2 e correspondencias com o ICD-10"
        )
        st.write(
            "O modelo foi treinado com a biblioteca Hugging Face *transformers* com base no modelo pré-treinado **bert-base-uncased** e está disponível em [https://huggingface.co/diogocarapito/text-to-icpc2](https://huggingface.co/diogocarapito/text-to-icpc2)"
        )
        st.write("Este projeto foi desenvolvido por Diogo Carapito com o apoio de bolsa de inviestigação da [AICIB](https://aicib.pt/) e [APMGF](https://apmgf.pt/) no âmbito do internato médico de MGF")

    # get available device
    available_device = device_cuda_mps_cpu(force_cpu=True)

    # model load and pipeline creation
    pipe = load_model("diogocarapito/text-to-icpc2", available_device)

    # text input
    text = st.text_input(
        "Coloque um diagnóstico para o modelo classificar", "diabetes mellitus"
    )

    # Record the start time
    start_time = timeit.default_timer()

    # Execute prediction
    prediction = pipe(text)

    # Record the end time
    end_time = timeit.default_timer()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # load validation dataset and create a list of corresponding codes and main description
    # label_list = load_val_dataset("diogocarapito/text-to-icpc2")

    # add the corresponding lable the corresponding description of the predicted code using val_dataset
    # add_labels_to_prediction(prediction, label_list)

    lables_dataframe = load_csv_github(
        "https://raw.githubusercontent.com/DiogoCarapito/text-to-icpc2/main/data/icpc2_processed.csv"
    )

    # Display the elapsed time
    st.write(
        f"Tempo necessário para classificação: **{elapsed_time:.4f} segundos** com **'{available_device}'**"
    )

    # Display the prediction
    prediction_display(prediction, lables_dataframe)


if __name__ == "__main__":
    main()
