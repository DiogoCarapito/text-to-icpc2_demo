import streamlit as st
from utils.utils import load_csv_github, load_predictions_labels
import pandas as pd
import numpy as np
import plotly.express as px

from datasets import load_dataset


def split():
    st.title("Train-Test Split")

    st.write("O dataset está bastenta imbalanceado e para alem disso ")

    st.write(
        "Durante o processo de treino há um split treino-teste ainda é relativamente básico"
    )
    st.markdown(
        """
        ```python
        dataset_split = dataset.train_test_split(
            test_size=0.2,
            stratify_by_column="label",
            seed=42
        )
        ```
        """
    )

    st.write(
        "Estas definições de split poderão ser refinadas, ou será necessário gerar mais dados para balancear o dataset, pois alugumas doenças têm poucos exemplos de treino."
    )

    st.divider()

    # load the frequency table
    df_frequency_model_result = load_csv_github(
        "https://raw.githubusercontent.com/DiogoCarapito/text-to-icpc2/main/data/data_pre_train.csv"
    )

    # count the frequency of each code
    frequency_table = df_frequency_model_result["code"].value_counts().reset_index()

    # merge the frequency table with the original dataset to get the chapter and origin
    frequency_table = df_frequency_model_result.merge(
        frequency_table, on="code", how="left"
    )

    # order by the code (index keeps the code order)
    frequency_table = frequency_table.sort_index()

    # load the correct predictions with a runid
    # runid is defined in the sidebar and is used to load the correct predictions
    df_correct_current_model = load_predictions_labels(st.session_state["runid"])

    # merge frequency_table with correct_prediction into df_frequency_model_result
    df_frequency_model_result = frequency_table.merge(
        df_correct_current_model, on="code", how="left"
    )

    # Assuming df_correct_current_model has a column indicating correctness, e.g., 'is_correct'
    df_frequency_model_result["is_correct"] = df_frequency_model_result[
        "is_correct"
    ].fillna(False)

    # create a version of the dataset with only the icpc2_description origin (only 726 codes)
    df_frequency_model_result_icpc2 = df_frequency_model_result[
        df_frequency_model_result["origin"] == "icpc2_description"
    ]

    # get the dataset from huggingface and split it like the training
    dataset_hf = load_dataset("diogocarapito/text-to-icpc2")

    # split the dataset into train and test as was done in the training
    dataset_after_split = dataset_hf["train"].train_test_split(
        test_size=0.2, stratify_by_column="label", seed=42
    )

    # get the test split and get all the labels that are included in the test split
    test_split_dataframe = dataset_after_split["test"].to_pandas()

    # create a list of codes that are present in the test split
    test_split_codes = test_split_dataframe["code"].values

    # get the labels that are included in the test split and put a true value in the column is_in_test_split
    df_frequency_model_result_icpc2[
        "is_in_test_split"
    ] = df_frequency_model_result_icpc2["code"].isin(test_split_codes)

    st.write(df_frequency_model_result_icpc2)
