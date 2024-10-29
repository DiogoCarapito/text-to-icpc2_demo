import streamlit as st
import timeit
from utils.utils import (
    device_cuda_mps_cpu,
    load_model,
    prediction_display,
    load_csv_github,
)

import pandas as pd

# import numpy as np
# import plotly.express as px


def demo_new():
    st.title("Demo")

    st.write(
        "O modelo está disponivel em [https://huggingface.co/diogocarapito/text-to-icpc2_bert-base-uncased](https://huggingface.co/diogocarapito/text-to-icpc2)"
    )

    # get available device
    available_device = device_cuda_mps_cpu(force_cpu=True)

    # choose model
    model_chosen = "text-to-icpc2_bert-base-uncased"

    pipe = load_model(f"diogocarapito/{model_chosen}", available_device)

    # text input
    text = st.text_input("Coloque um diagnóstico para o modelo classificar")

    # Record the start time
    start_time = timeit.default_timer()

    # Execute prediction
    predictions = pipe(text)

    # Record the end time
    end_time = timeit.default_timer()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Display the elapsed time
    st.write(
        f"Tempo necessário para classificação: **{elapsed_time:.4f} segundos** com **'{available_device}'**"
    )

    lables_dataframe = pd.read_csv(
        "https://raw.githubusercontent.com/DiogoCarapito/text-to-icpc2/main/data/icpc2_processed.csv"
    )

    lable_code_dict = pd.read_csv(
        "https://raw.githubusercontent.com/DiogoCarapito/text-to-icpc2/refs/heads/main/data/code_text_label.csv"
    )

    # st.write(lable_code_dict)

    for each in predictions:
        # transform the lable into a icpc2 code
        if each["label"].startswith("LABEL_"):
            # split on the "_" and get the 2nd part and make it integer
            each["label"] = int(each["label"].split("_")[1])

            # get the code
            each["code"] = lable_code_dict.loc[
                lable_code_dict["label"] == each["label"], "code"
            ].values[0]

            # get the desctição
            each["text"] = lable_code_dict.loc[
                lable_code_dict["label"] == each["label"], "text"
            ].values[0]

    # st.write(predictions)

    prediction_display(predictions, lables_dataframe)
