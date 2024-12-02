# Keeping tracks with the running times of each approach

**Please keep in mind that this is heavily subjected to specific hardware/software configurations, as well as concurrent running processes.**
NBA took round 500 seconds to build the index, which only needs to be done once.

- NBA without parallel: At least 3600s
- NBA with parallel: Around 1500s
- XGBoost: 65s
- Dot product: 11s for 10 predictions
