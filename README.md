# Next Best Action

Next best action recommendation engine recommends the next best action base on similar observations via an Annoy index.

There should be a trainng data file called data.csv
There should be a testing data file called data-val.parquet
To run other files you might need various dictionary files, check the code.

The relevant .pkl and .ann files are in `/results`.

Because Annoy doesn't support pickling, the index is saved to a separate .ann file. To load it, run:

```
from annoy import AnnoyIndex
import pickle

# Load the transformer and y_train
with open('collaborative_search_model.pkl', 'rb') as f:
    transformer, y_train = pickle.load(f)

# Load the Annoy index from its file
annoy_index = AnnoyIndex(X.shape[1], 'angular')  # Replace `X.shape[1]` with the correct feature dimension
annoy_index.load('result/annoy_index.ann')
```
