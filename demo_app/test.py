import joblib

# Load the saved dictionary
data = joblib.load('D://Documents/VCB/NBA/Real world use/results/collaborative_search_model_with_parallel.joblib')

# Access individual components
transformer = data['transformer']
y_train = data['y_train']
predict_nba_parallel = data['predict_nba_parallel']

# Example: Use the loaded function
# Assuming predict_nba_parallel requires inputs
result = predict_nba_parallel(y_train)  # Modify based on your actual function arguments

print(result)
