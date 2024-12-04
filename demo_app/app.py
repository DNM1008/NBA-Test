import dill
import numpy as np
import pandas as pd
import streamlit as st


# Load the saved objects
@st.cache_resource
def load_saved_model():
    with open("collaborative_search_model_with_parallel.pkl", "rb") as f:
        saved_objects = dill.load(f)
    return saved_objects


# Load the Annoy index (if required for prediction logic)
@st.cache_resource
def load_annoy_index():
    from annoy import AnnoyIndex

    f = 446  # Dimensionality, replace with the actual dimension of your index
    annoy_index = AnnoyIndex(f, "angular")
    annoy_index.load("annoy_index.ann")
    return annoy_index


# Age binning function
def labelling(df, col, bin_edges, labels):
    df["Age_group"] = pd.cut(
        df[col], bins=bin_edges, labels=labels, include_lowest=True, right=False
    )
    return df


# Initialize
saved_objects = load_saved_model()
transformer = saved_objects["transformer"]
predict_function = saved_objects["predict_nba_parallel"]

# Define the input fields
st.title("Loan Recommendation App")
st.markdown("Enter the values for the required variables to get a prediction.")

# Input all variables dynamically
variables = [
    "CBALQ_3m",
    "AVG_SL_SP_BOSUNG",
    "NO_TREN_CO_6m",
    "LOAIHINHCOQUANDANGCONGTAC",
    "SUM_CBALQ_LH",
    "BHNT_flag",
    "MEDIAN_GR_SUM_AMT",
    "TINHTRANGSOHUUNHA",
    "TINHTRANGHONNHAN",
    "Khu_vuc",
    "BHNT_after21",
    "Sum_PPC",
    "MEDIAN_GR_THGCO",
    "BHSK_remain",
    "IS_BANCAS",
    "AVG_GR_CBALQ",
    "CBALQ_6m",
    "AVG_CBALQ_6m",
    "BHNT_remain",
    "AVG_GR_THGCO",
    "IS_TM.1",
    "Age_y",
    "THGCO_3m",
    "CNT_TGCCKH",
    "THGNO_6m",
    "IS_TA",
    "TONGTHUNHAPHANGTHANG",
    "Snapshot",
    "BHSK_flag",
    "THGCO_6m",
    "MEDIAN_GR_CBALQ",
    "AVG_CBALQ_TGCCKH",
    "THGNO_3m",
    "TINHCHATCONGVIECHIENTAI",
    "AVG_AMT_3M",
    "NO_TREN_CO_3m",
    "AVG_CBALQ_3m",
    "Prio_flag",
    "SONGUOIPHUTHUOC",
    "THOIGIANLAMVIECLVHT",
    "BHSK_after21",
    "Payroll_Flag",
    "AVG_GR_THGNO",
    "CUS_GEN",
    "MEDIAN_GR_THGNO",
]

input_data = {}
for var in variables:
    input_value = st.text_input(f"Enter value for {var}:")
    # Default to "None" if no data is entered
    input_data[var] = input_value if input_value else "None"

# Predict button
if st.button("Predict"):
    # Convert input to DataFrame
    input_df = pd.DataFrame([input_data])

    # Convert Age_y to numeric and replace "None" with NaN
    input_df["Age_y"] = pd.to_numeric(input_df["Age_y"], errors="coerce")

    # Define age bins and labels
    age_bin_edges = [0, 20, 25, 30, 35, 40, 45, 50, 55, 60, float("inf")]
    age_labels = [
        "Duoi 20",
        "20 toi 24",
        "25 toi 29",
        "30 toi 34",
        "35 toi 39",
        "40 toi 44",
        "45 toi 49",
        "50 toi 54",
        "55 toi 59",
        "Tren 60",
    ]

    # Apply labelling function
    input_df = labelling(input_df, "Age_y", age_bin_edges, age_labels)

    # Apply transformations
    try:
        X_transformed = transformer.transform(input_df)
        X_transformed = X_transformed.astype(np.float32)  # Ensure proper dtype

        # Run prediction
        result = predict_function(X_transformed)
        st.success(f"Prediction Result: {result}")
        st.write(f"Age Group: {input_df['Age_group'].iloc[0]}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
