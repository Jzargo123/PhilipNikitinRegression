import pandas as pd
import os
from sklearn.model_selection import train_test_split


def read_data(data_path, split=False, fitting=True):
    """
    Reads data from the file in csv format with header in the first row, indexes in the first column,
    target in the last column. Also it splits  the dataset into train and test parts.

    :param data_path: str: path to the file
    :param split: boot: sptit dataset
    :return:
    """
    if os.path.exists(data_path):
        dataframe = pd.read_csv(data_path, index_col=0)
    else:
        raise ValueError("File {} doesn't exists".format(data_path))
    if not fitting:
        return dataframe.values
    original_x = dataframe.values[:, :-1]
    original_y = dataframe.values[:, -1]
    if not split:
        return original_x, original_y
    else:
        x_train, x_test, y_train, y_test = train_test_split(original_x, original_y)
        return x_train, x_test, y_train, y_test
