import streamlit as st
from transformers import pipeline
from datasets import load_dataset
import torch


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


@st.cache_data()
def load_val_dataset(dataset_link):
    # Load the dataset to get the corresponding codes
    dataset = load_dataset(dataset_link)

    # transform to pandas DataFrame
    dataset = dataset["train"].to_pandas()

    # filter only to origin icpc2_description
    val_dataset = dataset[dataset["origin"] == "icpc2_description"]

    # transform into a list
    val_list = val_dataset[["code", "text"]]

    return val_list


def add_labels_to_prediction(prediction, val_list):
    for each in prediction:
        # add a new key,value pair to each dict
        each["description"] = val_list[val_list["code"] == each["label"]][
            "text"
        ].values[0]
    return prediction


def prediction_display(prediction):
    for each in prediction:
        # st.metric(
        #     label="ICPC2 predicted code",
        #     value=f"{each['label']}",
        #     label_visibility="collapsed")

        st.write(f"## {each['label']} - {each['description']}")


def func():
    return None
