import streamlit as st
import timeit
from utils.utils import (
    device_cuda_mps_cpu,
    load_model,
    prediction_display,
    load_csv_github,
)

# import pandas as pd
# import numpy as np
# import plotly.express as px


def demo():
    st.title("Demo")

    st.write(
        "O modelo está disponivel em [https://huggingface.co/diogocarapito/text-to-icpc2](https://huggingface.co/diogocarapito/text-to-icpc2)"
    )

    # get available device
    available_device = device_cuda_mps_cpu(force_cpu=True)

    # model load and pipeline creation
    pipe = load_model("diogocarapito/text-to-icpc2", available_device)

    # text input
    text = st.text_input("Coloque um diagnóstico para o modelo classificar")

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
