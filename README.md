# Regression model


## Dependencies
 - sklearn
 - pandas
 - numpy
 - scipy
 
 More details is displayed in requirements.txt 

## Fit data
The tool provides simple way to train your own model using command line interface. 
File `train.py` is run with arguments:
- `data_path` path to csv file, be carefut that your file has a header in the first raw, 
indexes in the first column and targets in the last. Also only comma as a separator are available. 
- `save model` path where model will be saved after fitting your data
- `split` True if separation data on train and test part is required. 
- `evalute` If it's true $R^2$ coefficient will be calculated on test part after fitting

##### Example of usage
```bash
python train.py --data_path=../data/regression.csv --save_model=../data/regression.model --split=True --evaluate=True
```


## Predict labels
To predict labels use the script `predict.py`. It is run with arguments:
- `data_path` path to csv file, be carefut that your file has a header in the first raw, 
indexes in the first column. Also only comma as a separator are available. 
- `saved model` path to saved model
- `save_result` path where predictions will be saved

##### Example of usage
 ```bash
python predict.py --data_path ../data/regression_features.csv --saved_model ../data/regression.model --save_result ../data/predictions
```



