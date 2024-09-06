import streamlit as st
from transformers import pipeline
import torch
import pandas as pd

# from datasets import load_dataset


def device_cuda_mps_cpu(force_cpu=False):
    if force_cpu:
        device = "cpu"
    else:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    return device


@st.cache_resource()
def load_model(model_name, available_device):
    # Load model and create a text-classification pipeline
    # model_name_or_path = "diogocarapito/text-to-icpc2" # loading the medium size model trained only with K (cardiovascular) codes

    # prepare the pipeline
    pipe = pipeline(
        "text-classification",
        model=model_name,
        tokenizer="bert-base-uncased",
        device=available_device,
    )
    return pipe


# @st.cache_data()
# def load_val_dataset(dataset_link):
#     # Load the dataset to get the corresponding codes
#     dataset = load_dataset(dataset_link)

#     # transform to pandas DataFrame
#     dataset = dataset["train"].to_pandas()

#     # filter only to origin icpc2_description
#     val_dataset = dataset[dataset["origin"] == "icpc2_description"]

#     # transform into a list
#     val_list = val_dataset[["code", "text"]]

#     return val_list


@st.cache_data()
def load_csv_github(github_raw_url):
    df = pd.read_csv(github_raw_url)
    return df


# def add_labels_to_prediction(prediction, val_list):
#     for each in prediction:
#         # add a new key,value pair to each dict
#         each["description"] = val_list[val_list["code"] == each["label"]][
#             "text"
#         ].values[0]
#     return prediction


def prediction_display(prediction, labels_dataframe):
    for each in prediction:
        label = each["label"]
        description = labels_dataframe[labels_dataframe["cod"] == label]["nome"].values[
            0
        ]

        st.write(f"## {label} - {description}")

        include = labels_dataframe[labels_dataframe["cod"] == label]["incl"].values[0]

        st.write("### Inclui")
        st.write(include)

        exclude = labels_dataframe[labels_dataframe["cod"] == label]["excl"].values[0]

        st.write("### Exclui")
        st.write(exclude)

        criteria = labels_dataframe[labels_dataframe["cod"] == label]["crit"].values[0]

        st.write("### Critérios")
        st.write(criteria)


def func():
    return None
