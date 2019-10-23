def df_to_arrays(df, y_column, columns_to_exclude, return_index=False):
    """Convert data frame to numpy array and split X and y"""

    # Remove missing rows with missing entries in y
    preprocessed = df.dropna(subset=[y_column])

    # Split to x and y
    y = preprocessed[y_column]
    X = preprocessed.drop(columns=list(columns_to_exclude + (y_column,)))

    # Get array representation
    X_arr = X.values.copy()
    y_arr = y.values.copy()

    # Check if needed to return index
    if return_index:
        name_to_idx = dict([(feature, X.columns.get_loc(feature)) for feature in X.columns])
        return X_arr, y_arr, name_to_idx
    else:
        return X_arr, y_arr