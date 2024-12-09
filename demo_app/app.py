import streamlit as st
import pandas as pd
from annoy import AnnoyIndex
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from concurrent.futures import ThreadPoolExecutor
import pickle

# App Title
st.title("BANCAS Marketing suggestion")
# Step 1: Read the dimension from the text file
with open("annoy_index_dim2.txt", "r") as file:
    dimension = int(file.read().strip())  # Read and convert to integer

# Step 2: Load the Annoy index using the dimension
index_file = "annoy_index2.ann"
annoy_index = AnnoyIndex(dimension, 'angular')  # Replace 'angular' with the actual metric used
annoy_index.load(index_file)

# Load the cleaned dataframe
train_df = pd.read_csv('cleaned_train.csv')
y_train = train_df['IS_BANCAS']

# Load the transformer back when needed
with open("transformer.pkl", "rb") as file:
    loaded_transformer = pickle.load(file)


num_neighbors = 20
# Define options for the dropdown menus
dropdown_options = {
    # "Age Group": ["18-25", "26-35", "36-45", "46-60", "60+"],
    "CUS_GEN": ["M", "F", "Other"],
    "LOAIHINHCOQUANDANGCONGTAC": ["Other", "Tổng công ty/Tập đoàn kinh tế Nhà nước; Các ngân hàng thương mại, công ty bảo hiểm, công ty đa quốc gia, tổ chức phi chính phủ, Doanh nghiệp FDI", "Các tổ chức, doanh nghiệp, định chế tài chính khác", "Các trường hợp khác (bao gồm cá nhân tự kinh doanh)", "Đơn vị vũ trang nhân dân", "Tổ chức, doanh nghiệp có XHTD là BB+, BB, B+", "Cơ quan Đảng, Cơ quan Nhà nước  tại địa phương", "CSKD trực thuộc CQ Đảng, CQ NN, TC CT-XH còn lại (tính đến cấp Quận, Huyện)", "Tổ chức, doanh nghiệp có XHTD là AAA, AA+, AA, A+", "Các định chế tài chính có XHTD là AAA, AA, A+, A, BBB, BB+", "Tổ chức, doanh nghiệp có XHTD là A, BBB", "Đơn vị sự nghiệp công lập tại Trung ương, Hà Nội, TP HCM (chỉ tính các quận)", "Cơ quan Đảng, Cơ quan Nhà nước  tại Trung Ương; Hà Nội , TP HCM (chỉ tính các quận)", "Đơn vị sự nghiệp công lập tại địa phương", "Tổng công ty/Tập đoàn kinh tế Nhà nước; Các ngân hàng thương mại, công ty bảo hiểm, công ty đa quốc ", "Cơ quan Đảng, Cơ quan Nhà nước  tại Trung Ương, Hà Nội , TP HCM (chỉ tính các quận)", "Tổ chức chính trị - xã hội - nghề nghiệp tại địa phương", "Tổ chức, doanh nghiệp có XHTD là C, D", "Đơn vị sự nghiệp ngoài công lập tại địa phương", "Các định chế tài chính có XHTD là BB, B+, B, CCC", "Tổ chức chính trị - xã hội - nghề nghiệp tại trung ương, Hà Nội, TP HCM (chỉ tính các quận)", "Các định chế tài chính có XHTD là CC+, CC, C+, C", "Đơn vị sự nghiệp ngoài công lập tại Trung ương, Hà Nội, TP HCM (chỉ tính các quận)"],
    "TINHTRANGSOHUUNHA":[ "Other", "Nhà đi thuê", "Nhà sở hữu riêng", "Ở chung nhà bố mẹ (trừ trường hợp bố mẹ cũng đi thuê nhà)", "Khác"],
    "TINHTRANGHONNHAN": ["Other", "Độc thân", "Có gia đình", "Ly thân hoặc đang trong quá trình giải quyết ly dị", "Ly dị hoặc góa"],
    "Khu_vuc": ["Tp.HCM", "HN", "BTTB", "ĐNB", "ĐBSH", "TNB", "NTBTN", "TDMNPB", "TSC"],
    "BHNT_after21": ["No Info", "No", "Yes"],
    "BHSK_after21": ["No Info", "Yes", "No"],
    "BHSK_remain": ["No Info", "No", "Yes"],
    "SONGUOIPHUTHUOC": ["Other", "Dưới 2 người", "2 người", "4 người", "3 người", "4  người"],
    "CUS_GEN": ["F", "M", " "],
    "BHNT_remain": ["No Info", "No", "Yes"],
    "TONGTHUNHAPHANGTHANG": ["None applicable", "22tr toi duoi 35 tr", "Duoi 8tr8", "Tren 100tr", "11tr5 toi duoi 15tr", "35tr toi duoi 50 tr", "15tr toi duoi 22tr", "8tr8 toi duoi 11tr5", "50 tr toi duoi 100 tr"],
    "So_du": ["80 toi 120 trieu", "Duoi 15 trieu", "None applicable", "15 toi duoi 43 trieu", "Tren 820 trieu", "120 toi 200 trieu", "360 toi 820 trieu", "43 toi duoi 80 trieu", "200 toi duoi 360 trieu"],
    # "IS_BANCAS": ["1.0", "0.0"],
    "Age_group": ["20 toi 24", "25 toi 29", "30 toi 34", "55 toi 59", "45 toi 49", "40 toi 44", "35 toi 39", "50 toi 54", "Tren 60", "Duoi 20"],
    
}

# Create empty dictionary to store user inputs
user_inputs = {}

# Generate dropdown menus
for key, options in dropdown_options.items():
    user_inputs[key] = st.selectbox(f"Select {key}", options)
## Collaborative search

## Collaborative search

# Step 1: Handle missing values and ensure consistent data types
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

with open("transformer.pkl", "rb") as file:
    transformer = pickle.load(file)
# Show a button to create the DataFrame
if st.button("Predict BANCAS"):
    # Convert inputs to a DataFrame
    user_df = pd.DataFrame([user_inputs])
    user_df['IS_BANCAS'] = 0
    # vector = user_df.iloc[0].tolist()
    target_column = 'IS_BANCAS'
    y_pred_col_train = predict_nba_parallel(user_df, target_column, annoy_index, transformer, y_train)
    y_pred_col_train_round = y_pred_col_train.round()
    # neighbor_indices = annoy_index.get_nns_by_vector(vector, num_neighbors)
    # similar_observations = train_df.iloc[neighbor_indices]
    
    st.write("Customer's expected BANCAS:")
    st.dataframe(y_pred_col_train)

    # Save the DataFrame to a CSV file (optional)
    # user_df.to_csv("user_inputs.csv", index=False)
    # st.success("DataFrame saved as user_inputs.csv!")
    
