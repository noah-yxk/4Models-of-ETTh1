# Transformer model
Code for Transformer
### Install 
```
pip install -r requirement.txt
```

### Data Processing
Place the dataset(train, valid, test) in the 'data' directory, and run(take test set as an example):
```
python time_series_forecasting/data_utils.py --csv_path "data/test_set.csv" --out_path "data/processed_test_data.csv" --config_path "data/config.json"
```

### Run
Train and test model: 
```
python run.py
```
