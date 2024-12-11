# Next Best Action

Next best action recommendation engine recommends the next best action base on similar observations via an Annoy index.

There should be a trainng data file called data.csv
There should be a testing data file called data-val.parquet
To run other files you might need various dictionary files, check the code.

The relevant .pkl and .ann files are in `/results`.

Because Annoy doesn't support pickling, the index is saved to a separate .ann file. To load it, run:

```
# Load the saved model and function
with open('results/collaborative_search_model_with_parallel.pkl', 'rb') as f: # or collaborative_search_model with_joblib.pkl
    model_components = dill.load(f)

# Access components
annoy_index = model_components['annoy_index']
transformer = model_components['transformer']
y_train = model_components['y_train']
predict_nba_parallel = model_components['predict_nba_parallel']  # Loaded function (or predict_nba_joblib depending on what you want)
```
