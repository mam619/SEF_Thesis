def shift(df, columns, lag_num, prefix=None, drop=None):
    """Shift and substitute columns in df with the provided lag_num.

    Args:
        df (pd.DataFrame): data frame with all the data
        columns (list): columns list to make the shift
        lag_num (int): number of rows to lag the columns
        prefix (str, optional): prefix to be added to the columns name. Defaults to None.

    Return:
        df(pd.DataFrame): data frame with data shifted
    """
    for i in columns:
        if prefix:
            df[f"{prefix}_{i}"] = df[i].shift(lag_num)
        else:
            df[i] = df[i].shift(lag_num)
            if drop is not None:
                df.drop(i, axis=1, inplace=True)

    return df
