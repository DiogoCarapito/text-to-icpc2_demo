import streamlit as st

# from transformers import pipeline
# from datasets import load_dataset
# import numpy as np
# import torch
import timeit

from utils.utils import (
    device_cuda_mps_cpu,
    load_model,
    load_val_dataset,
    add_labels_to_prediction,
    prediction_display,
)


def main():
    st.title("text-to-icpc2 Demo")

    # get available device
    available_device = device_cuda_mps_cpu(force_cpu=True)

    # model load and pipeline creation
    pipe = load_model("diogocarapito/text-to-icpc2", available_device)

    # text input
    text = st.text_input("Coloque um diagnóstico para codificação", "diabetes mellitus")

    # Record the start time
    start_time = timeit.default_timer()

    # Execute prediction
    prediction = pipe(text)

    # Record the end time
    end_time = timeit.default_timer()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # load validation dataset and create a list of corresponding codes and main description
    label_list = load_val_dataset("diogocarapito/text-to-icpc2")

    # add the corresponding lable the corresponding description of the predicted code using val_dataset
    add_labels_to_prediction(prediction, label_list)

    # Display the prediction
    prediction_display(prediction)

    # Display the elapsed time
    st.write(
        f"Time taken to classify: {elapsed_time:.4f} seconds using {available_device}"
    )


if __name__ == "__main__":
    main()
