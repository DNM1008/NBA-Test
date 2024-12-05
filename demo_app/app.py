from annoy import AnnoyIndex
import numpy as np
import pandas as pd
import streamlit as st
import pickle
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer





# Load the Annoy index (if required for prediction logic)

# def load_annoy_index():
#     from annoy import AnnoyIndex

#     f = 446  # Dimensionality, replace with the actual dimension of your index
#     annoy_index = AnnoyIndex(f, "angular")
#     annoy_index.load("annoy_index.ann")
#     return annoy_index


# # Age binning function
# def labelling(df, col, bin_edges, labels):
#     df["Age_group"] = pd.cut(
#         df[col], bins=bin_edges, labels=labels, include_lowest=True, right=False
#     )

# Example: Use the loaded function
# Assuming predict_nba_parallel requires inputs
index = AnnoyIndex(446, 'angular')
index.load("annoy_index.ann")
dictionary_path = 'bin_edges_dict.pkl'

with open (dictionary_path, 'rb') as file:
    bin_edges_dict = pickle.load(file)
print(type(bin_edges_dict)) 


# # Initialize
# saved_objects = joblib.load('D://Documents/VCB/NBA/Real world use/results/collaborative_search_model_with_parallel.joblib')
# transformer = saved_objects["transformer"]
# y_train = saved_objects['y_train']
# predict_function = saved_objects["predict_nba_parallel"]

# Define the input fields
st.title("Loan BANCAS App")
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
    input_df['Age_group'] = pd.cut(input_df['Age_y'], bins=age_bin_edges, labels=age_labels, right=False)

    
    def apply_labelling_with_dict(df, bin_edges_dict):
        """
        Applies binning to a DataFrame using precomputed bin edges.

        Parameters:
        - df: DataFrame to label.
        - bin_edges_dict: Dictionary containing precomputed bin edges.

        Returns:
        - df: Updated DataFrame with consistent binned columns.
        """
        for col, bin_edges in bin_edges_dict.items():
            if col in df.columns:
                # Apply consistent binning
                df[col] = pd.cut(df[col], bins=bin_edges, labels=False, right=False)
            else:
                print(f"Warning: Column '{col}' is not present in the dataset.")
        return df



    def predict_nba_parallel(df, target_column, annoy_index, transformer, y_train, n_neighbors=20, n_jobs=4):
    # Preprocess the data and separate features
        X, y, _ = preprocess_data(df, target_column, transformer)
        
        # Function to process a single observation
        def process_vector(test_vector):
            nearest_neighbors = annoy_index.get_nns_by_vector(test_vector, n_neighbors)  # Find neighbors
            similar_train_data = y_train.iloc[nearest_neighbors]  # Use y_train directly
            return similar_train_data.mean()  # Average for prediction
        
        # Parallelize using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            y_pred = list(executor.map(process_vector, X))
        
        return np.array(y_pred)
    def handle_missing_values(df):
        df = df.astype(str)
        df = df.fillna("None")
        return df
    # Step 2: Preprocess - Encoding categorical variables
    def preprocess_data(df, target_column, transformer=None):
        # Separate features (X) and target (y)
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Reset indices for alignment with Annoy
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        
        # Handle missing values and ensure consistent data types
        X = handle_missing_values(X)

        if transformer is None:
            transformer = ColumnTransformer(
                transformers=[
                    ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), X.columns.tolist())
                ],
                remainder='passthrough'
            )
            transformer.fit(X)
        
        # Transform features
        X_transformed = transformer.transform(X)
        return X_transformed.astype(np.float32), y.astype(float), transformer

    cols_to_label = ['AVG_AMT_3M', 'AVG_CBALQ_3m', 'AVG_CBALQ_6m', 'AVG_CBALQ_TGCCKH',
       'AVG_GR_CBALQ', 'AVG_GR_THGCO', 'AVG_GR_THGNO', 'AVG_SL_SP_BOSUNG',
        'CBALQ_3m', 'CBALQ_6m', 'CNT_TGCCKH', 'MEDIAN_GR_CBALQ',
       'MEDIAN_GR_SUM_AMT', 'MEDIAN_GR_THGCO', 'MEDIAN_GR_THGNO',
       'NO_TREN_CO_3m', 'NO_TREN_CO_6m', 'SUM_CBALQ_LH', 'Snapshot', 'Sum_PPC',
       'THGCO_3m', 'THGCO_6m', 'THGNO_3m', 'THGNO_6m', 'TONGTHUNHAPHANGTHANG']

    input_df =apply_labelling_with_dict(input_df, bin_edges_dict)

    # Apply transformations
    # try:
    #     X_transformed = transformer.transform(input_df)
    #     X_transformed = X_transformed.astype(np.float32)  # Ensure proper dtype

    #     # Run prediction
    result = predict_nba_parallel(input_df)
    st.success(f"Prediction Result: {result}")
    st.write(f"Age Group: {input_df['Age_group'].iloc[0]}")
        

