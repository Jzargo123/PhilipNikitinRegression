"""
Main module for fitting the model
"""

import argparse

from sklearn.metrics import r2_score
from Regression.regression import LinearRegression
from Regression.preprocessing import read_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', action='store', help='Path to csv file with data')
    parser.add_argument('--save_model', action='store', help='Path where to save model')
    parser.add_argument('--split', type=bool, default=False, help='Split data')
    parser.add_argument('--evaluate', type=bool, default=False, help='Evaluate the model. Must use only with split')
    args = parser.parse_args()
    print(args)
    if args.split:
        train_x, test_x, train_y, test_y = read_data(args.data_path, split=True)
    else:
        train_x, train_y = read_data(args.data_path, split=True)
    model = LinearRegression(reg_l1=0., reg_l2=0.)
    model.fit(train_x, train_y)
    model.save_model(args.save_model)

    if args.split and args.evaluate:
        print(r2_score(test_y, model.predict(test_x)))


if __name__ == "__main__":
    main()

