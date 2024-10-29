import streamlit as st
import timeit
from utils.utils import (
    device_cuda_mps_cpu,
    load_model,
    prediction_display,
    load_csv_github,
)
import pyperclip
import pandas as pd

import os
from dotenv import load_dotenv
from supabase import create_client, Client
from datetime import datetime


def load_supabase():
    # load .env file
    load_dotenv()

    # get Supabase URL and Key from environment variables
    url: str = os.environ.get("SUPABASE_URL")
    key: str = os.environ.get("SUPABASE_KEY")
    supabase_client: Client = create_client(url, key)

    return supabase_client


# Load Supabase
supabase = load_supabase()


# Function to insert data into Supabase
def supabase_insert(
    text_input, predicted_text, predicted_code, predicted_lable, model, feedback, copy
):
    # Get current datetime
    date_time = datetime.now().isoformat()

    # Create the data in a format to be inserted into Supabase
    sb_insert = {
        "created_at": date_time,
        "text_input": text_input,
        "predicted_text": predicted_text,
        "predicted_code": predicted_code,
        "predicted_lable": predicted_lable,
        "model": model,
        "feedback": feedback,
        "copy": copy,
    }

    # Insert data into Supabase
    supabase.table("demo_text-to-icpc2").insert(sb_insert).execute()


def demo_new():
    st.header("Demo")

    # st.write(
    #     "O modelo está disponivel em [https://huggingface.co/diogocarapito/text-to-icpc2_bert-base-uncased](https://huggingface.co/diogocarapito/text-to-icpc2)"
    # )

    # get available device
    available_device = device_cuda_mps_cpu(force_cpu=True)

    # choose model
    model_chosen = "text-to-icpc2_bert-base-uncased"

    pipe = load_model(f"diogocarapito/{model_chosen}", available_device)

    # text input
    text = st.text_input("Coloque um diagnóstico para o modelo classificar")

    if text != "":
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

        if "copy" not in st.session_state:
            st.session_state["copy"] = False

        if "feedback" not in st.session_state:
            st.session_state["feedback"] = None

        st.write("")
        st.subheader("O modelo portou-se bem?")
        
        col_1, col_2, col_3 = st.columns([1,1,1])

        with col_1:
            if st.button("Copiar código", type="primary"):
                pyperclip.copy(predictions[0]["code"])
                st.session_state["copy"] = True

        with col_2:
            if st.button("👍", type="primary"):
                st.session_state["feedback"] = 1

        with col_3:
            if st.button("👎", type="secondary"):
                st.session_state["feedback"] = -1
        st.write("")
        # text_input, predicted_code, predicted_lable, model
        prediction_display(predictions, lables_dataframe)
        
        supabase_insert(
            text,  # text_input
            predictions[0]["text"],  # predicted_text
            predictions[0]["code"],  # predicted_code
            predictions[0]["label"],  # predicted_lable
            model_chosen,  # model
            st.session_state["feedback"],
            st.session_state["copy"],
        )

    else:
        st.warning("Coloque um diagnóstico para classificar")
