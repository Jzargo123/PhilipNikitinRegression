"""
Main part for prediction
"""
import argparse
import numpy as np

from Regression.regression import LinearRegression
from Regression.preprocessing import read_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', action='store', help='Path to csv file with data')
    parser.add_argument('--saved_model', action='store', help='Path where to save model')
    parser.add_argument('--save_result', action='store', help='Path where to save result')
    args = parser.parse_args()

    features = read_data(args.data_path, split=False, fitting=False)
    model = LinearRegression()
    model.load_model(args.saved_model)
    predictions = model.predict(features)
    np.save(args.save_result, predictions)


if __name__ == "__main__":
    main()

