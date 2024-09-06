import streamlit as st
from utils.utils import load_csv_github
import pandas as pd
import numpy as np
import plotly.express as px


def validation():
    st.title("Validation")
    st.write("This page is under construction")


# frequency_table = load_csv_github("https://raw.githubusercontent.com/DiogoCarapito/text-to-icpc2/main/data/data_pre_train.csv")

# # count the frequency of each code
# frequency_table = frequency_table["code"].value_counts()

# # order by "code"
# frequency_table = frequency_table.sort_values("code", axis=0).reset_index()


# #frequency_table['count'] = frequency_table['code'].apply(lambda x: np.log2(x))

# st.write("frequency_table")
# st.write(frequency_table)

# #df_correct_current_model = load_csv_github("https://raw.githubusercontent.com/DiogoCarapito/text-to-icpc2/main/correct_predictions/correct_predictions_862e53bb1e7a4c05ab8a049c5a97a257.csv")
# df_correct_current_model = pd.read_csv("https://raw.githubusercontent.com/DiogoCarapito/text-to-icpc2/main/correct_predictions/correct_predictions_862e53bb1e7a4c05ab8a049c5a97a257.csv")

# # Create a list of codes that are present in correct_prediction
# df_correct_current_model = df_correct_current_model[["code","top_prediction"]]

# df_correct_current_model = df_correct_current_model.rename(columns={"top_prediction": "is_correct"})
# df_correct_current_model["is_correct"] = True

# st.write("df_correct_current_model")
# st.write(df_correct_current_model)

# # merge frequency_table with correct_prediction
# frequency_table = frequency_table.merge(df_correct_current_model, on="code", how="left")

# # Assuming df_correct_current_model has a column indicating correctness, e.g., 'is_correct'
# frequency_table["is_correct"] = frequency_table["is_correct"].fillna(False)

# # add a sqrt count to the frequency_table["count"] to make visualization easier
# frequency_table["count_log2"] = np.log2(frequency_table["count"])#.astype(int)

# # metric with the number of correct predictions and percentage
# col_1_1, col_1_2 = st.columns(2)
# num_correct = frequency_table[frequency_table["is_correct"] == True].shape[0]
# num_codes = frequency_table.shape[0]
# percentage_correct = round(100 * num_correct / num_codes, 2)
# with col_1_1:
#     st.metric("Número de previsões corretas", num_correct)
# with col_1_2:
#     st.metric("Porcentagem de previsões corretas", f"{percentage_correct}%")


# # Create a bar chart with Plotly
# fig = px.bar(
#     frequency_table,
#     x="code",
#     y="count_log2",
#     color="is_correct",
#     color_discrete_map={True: "green", False: "red"},
#     title="Frequency of Codes",
# )

# # Sort the x-axis alphabetically
# fig.update_layout(xaxis={"categoryorder": "category ascending"})


# # Display the Plotly chart in Streamlit
# st.plotly_chart(fig)

# st.write(df_correct_current_model)
