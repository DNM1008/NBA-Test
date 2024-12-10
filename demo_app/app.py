import streamlit as st
import pandas as pd
from annoy import AnnoyIndex
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from concurrent.futures import ThreadPoolExecutor
import pickle
import xgboost as xgb

# App Title
st.title("BANCAS Marketing Suggestion")

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
template = pd.read_csv('input.csv')


# Load the transformer
with open("transformer.pkl", "rb") as file:
    loaded_transformer = pickle.load(file)
# Load the xgb model
with open('xgb_model_simplified.pkl', 'rb') as file:
    xgb_model = pickle.load(file)



# Number of neighbors
num_neighbors = 20

# Define options for dropdown menus (without sorting)
demographic_options = {
    "Age_group": ["Duoi 20", "20 toi 24", "25 toi 29", "30 toi 34", "55 toi 59", "45 toi 49", "40 toi 44", "35 toi 39", "50 toi 54", "Tren 60"],
    "CUS_GEN": ["M", "F", "Other"],
    "Khu_vuc": ["Tp.HCM", "HN", "BTTB", "ĐNB", "ĐBSH", "TNB", "NTBTN", "TDMNPB", "TSC"],
    "SONGUOIPHUTHUOC": ["Other", "Dưới 2 người", "2 người", "4 người", "3 người", "4  người"],
    "TINHTRANGHONNHAN": ["Other", "Độc thân", "Có gia đình", "Ly thân hoặc đang trong quá trình giải quyết ly dị", "Ly dị hoặc góa"],
    "TINHTRANGSOHUUNHA": ["Other", "Nhà đi thuê", "Nhà sở hữu riêng", "Ở chung nhà bố mẹ (trừ trường hợp bố mẹ cũng đi thuê nhà)", "Khác"],
    "LOAIHINHCOQUANDANGCONGTAC": [
                                  "Tổng công ty/Tập đoàn kinh tế Nhà nước; Các ngân hàng thương mại, công ty bảo hiểm, công ty đa quốc gia, tổ chức phi chính phủ, Doanh nghiệp FDI",
                                  "Đơn vị sự nghiệp công lập tại Trung ương, Hà Nội, TP HCM (chỉ tính các quận)", 
                                  "Cơ quan Đảng, Cơ quan Nhà nước tại Trung Ương, Hà Nội , TP HCM (chỉ tính các quận)", 
                                  "Tổ chức chính trị - xã hội - nghề nghiệp tại địa phương", 
                                  "Đơn vị sự nghiệp công lập tại địa phương", 
                                  "Đơn vị vũ trang nhân dân", 
                                  "Các định chế tài chính có XHTD là AAA, AA, A+, A, BBB, BB+", 
                                  "Tổ chức, doanh nghiệp có XHTD là AAA, AA+, AA, A+", 
                                  "Tổ chức, doanh nghiệp có XHTD là A, BBB", 
                                  "Tổ chức, doanh nghiệp có XHTD là BB+, BB, B+", 
                                  "Các trường hợp khác (bao gồm cá nhân tự kinh doanh)", 
                                  "Các định chế tài chính có XHTD là BB, B+, B, CCC", 
                                  "Các định chế tài chính có XHTD là CC+, CC, C+, C", 
                                  "Tổ chức, doanh nghiệp có XHTD là C, D", 
                                  "Các tổ chức, doanh nghiệp, định chế tài chính khác", 
                                  "Cơ quan Đảng, Cơ quan Nhà nước tại địa phương", 
                                  "CSKD trực thuộc CQ Đảng, CQ NN, TC CT-XH còn lại (tính đến cấp Quận, Huyện)", 
                                  "Cơ quan Đảng, Cơ quan Nhà nước tại Trung Ương; Hà Nội , TP HCM (chỉ tính các quận)", 
                                  "Tổng công ty/Tập đoàn kinh tế Nhà nước; Các ngân hàng thương mại, công ty bảo hiểm, công ty đa quốc", 
                                  "Đơn vị sự nghiệp ngoài công lập tại địa phương", 
                                  "Tổ chức chính trị - xã hội - nghề nghiệp tại trung ương, Hà Nội, TP HCM (chỉ tính các quận)", 
                                  "Đơn vị sự nghiệp ngoài công lập tại Trung Ương, Hà Nội, TP HCM (chỉ tính các quận)",
                                  "Other"], 
}

casa_options = {
    "BHNT_after21": ["No Info", "No", "Yes"],
    "BHSK_after21": ["No Info", "No", "Yes"],
    "BHSK_remain": ["No Info", "No", "Yes"],
    "BHNT_remain": ["No Info", "No", "Yes"],
    "TONGTHUNHAPHANGTHANG": [
        "Duoi 8tr8", 
        "8tr8 toi duoi 11tr5", 
        "11tr5 toi duoi 15tr", 
        "15tr toi duoi 22tr", 
        "22tr toi duoi 35 tr", 
        "35tr toi duoi 50 tr", 
        "50 tr toi duoi 100 tr",
        "Tren 100tr",
        "None applicable", 
        ],
    "So_du": [
        "Duoi 15 trieu", 
        "15 toi duoi 43 trieu", 
        "43 toi duoi 80 trieu", 
        "80 toi 120 trieu", 
        "120 toi 200 trieu", 
        "200 toi duoi 360 trieu",
        "360 toi 820 trieu", 
        "Tren 820 trieu", 
        "None applicable", 
        ],
}

# Create empty dictionary to store user inputs
user_inputs = {}

# Generate Demographic Variables section
st.header("Demographic Variables")
for key, options in demographic_options.items():
    user_inputs[key] = st.selectbox(f"Select {key}", options)

# Generate CASA Variables section
st.header("CASA Variables")
for key, options in casa_options.items():
    user_inputs[key] = st.selectbox(f"Select {key}", options)

# Define preprocessing and prediction functions (same as original code)
def handle_missing_values(df):
    df = df.astype(str)
    df = df.fillna("None")
    return df

def preprocess_data(df, target_column, transformer=None):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X = handle_missing_values(X)

    if transformer is None:
        transformer = ColumnTransformer(
            transformers=[('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), X.columns.tolist())],
            remainder='passthrough'
        )
        transformer.fit(X)

    X_transformed = transformer.transform(X)
    return X_transformed.astype(np.float32), y.astype(float), transformer

def predict_nba_parallel(df, target_column, annoy_index, transformer, y_train, n_neighbors=20, n_jobs=4):
    X, y, _ = preprocess_data(df, target_column, transformer)

    def process_vector(test_vector):
        nearest_neighbors = annoy_index.get_nns_by_vector(test_vector, n_neighbors)
        similar_train_data = y_train.iloc[nearest_neighbors]
        return similar_train_data.mean()

    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        y_pred = list(executor.map(process_vector, X))

    return np.array(y_pred)


with open("transformer.pkl", "rb") as file:
    transformer = pickle.load(file)
# Prediction button
if st.button("Predict BANCAS"):
    # Convert inputs to a DataFrame
    user_df = pd.DataFrame([user_inputs])
    user_df_xgb = user_df.astype('category')
    user_df['IS_BANCAS'] = 0
    
    booster = xgb_model.get_booster()
    feature_names = booster.feature_names
    user_df_xgb = user_df_xgb[sorted(user_df_xgb.columns)]    
    user_df_xgb = user_df_xgb[feature_names]
    user_df_xgb.to_csv('input.csv')
    
    
    target_column = 'IS_BANCAS'
    y_pred_col_train = predict_nba_parallel(user_df, target_column, annoy_index, transformer, y_train)
    y_pred_col_train_round = y_pred_col_train.round()
    
    
    # neighbor_indices = annoy_index.get_nns_by_vector(vector, num_neighbors)
    # similar_observations = train_df.iloc[neighbor_indices]
    y_pred_xgb = xgb_model.predict_proba(user_df_xgb)[:,1]
    
    st.write("Customer's expected BANCAS (Annoy Index):")
    st.write(y_pred_col_train)
    # st.dataframe(user_df_xgb)
    

    st.write("Customer's expected BANCAS (XGBoost):")
    st.write(y_pred_xgb)
    st.write(user_df_xgb)
    # st.dataframe(feature_names)
    # st.dataframe(template)

    # Save the DataFrame to a CSV file (optional)
    # user_df.to_csv("user_inputs.csv", index=False)
    # st.success("DataFrame saved as user_inputs.csv!")
    
