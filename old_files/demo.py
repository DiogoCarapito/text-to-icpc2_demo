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

    # model load and pipe_1line creation

    # choose model
    model_chosen_1 = "text-to-icpc2"
    model_chosen_2 = "text-to-icpc2_bert-base-uncased"

    pipe_1 = load_model(f"diogocarapito/{model_chosen_1}", available_device)
    pipe_2 = load_model(f"diogocarapito/{model_chosen_2}", available_device)

    # text input
    text = st.text_input("Coloque um diagnóstico para o modelo classificar")

    # Record the start time
    start_time_1 = timeit.default_timer()

    # Execute prediction
    prediction_1 = pipe_1(text)

    # Record the end time
    end_time_1 = timeit.default_timer()

    # Calculate the elapsed time
    elapsed_time_1 = end_time_1 - start_time_1

    label_code_dict = load_csv_github(
        "https://raw.githubusercontent.com/DiogoCarapito/text-to-icpc2/refs/heads/main/data/code_text_label.csv"
    )

    # remove LABEL_ from the label and get the code and text if the label starts with LABEL_
    # transform the label into a icpc2 code
    for each in prediction:
        if each["label"].startswith("LABEL_"):
            label = each["label"].split("_")[1]
            # make label an integer
            label = int(label)

            # print the code and the text matching the label
            each["label"] = label_code_dict[label_code_dict["label"] == label]

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

    label_code_dict = load_csv_github(
        "https://raw.githubusercontent.com/DiogoCarapito/text-to-icpc2/refs/heads/main/data/code_text_label.csv"
    )

    # get the text and code of the prediction by matching the label with the code

    # for each in prediction:
    #     #extract code given the label: "LABEL_474" into "474"
    #     label = each["label"]
    #     label = label.split("_")[1]

    #     # make label an integer
    #     label = int(label)

    #     # print the code and the text matching the label
    #     code_text = label_code_dict[label_code_dict["label"] == label]

    #     st.write(code_text["code"].values[0])
    #     st.write(code_text["text"].values[0])

    # st.write(lables_dataframe)

    # Display the prediction
    st.write("Diagnóstico: ", prediction)
    prediction_display(code_text, lables_dataframe)
