import json
import random

import numpy as np
import pandas as pd
from tqdm import tqdm
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler

from time_series_forecasting.model import TimeSeriesForcasting


def split_df(
    df: pd.DataFrame, split: str, history_size: int = 96, horizon_size: int = 96
):
    """
    Create a training / validation samples
    Validation samples are the last horizon_size rows

    :param df:
    :param split:
    :param history_size:
    :param horizon_size:
    :return:
    """

    end_index = df.shape[0]
    label_index = end_index - horizon_size
    start_index = max(0, label_index - history_size)

    history = df[start_index:label_index]
    targets = df[label_index:end_index]

    return history, targets


def pad_arr(arr: np.ndarray, expected_size: int = 120):
    """
    Pad top of array when there is not enough history
    :param arr:
    :param expected_size:
    :return:
    """
    arr = np.pad(arr, [(expected_size - arr.shape[0], 0), (0, 0)], mode="edge")
    return arr


def df_to_np(df):
    arr = np.array(df)
    # arr = pad_arr(arr)
    return arr

def get_groups(data, window_size):
    
    new_data = pd.DataFrame(columns=data.columns)
    window_list = []
    # 构建滑动窗口
    for i in tqdm(range((len(data) - window_size + 1)), desc="Processing"):
        window_data = data.loc[i:i+window_size-1, :].copy()
        window_data['group'] = i
        window_list.append(window_data)
    new_data = pd.concat(window_list, ignore_index=True)
    
    grp_by_train = new_data.groupby(by='group')
    groups = list(grp_by_train.groups)

    return grp_by_train, groups



def normalize_data(data: pd.DataFrame, target_titles):
    data_normalized = data.copy()
    scalers = []
    for target_title in target_titles:
        scaler = MinMaxScaler()
        temp = data_normalized[target_title].values
        temp = temp.reshape((len(temp), 1))


        temp = scaler.fit_transform(temp)
        temp = temp.reshape((len(temp)))
        data_normalized[target_title] = temp
        scalers.append(scaler)
    return data_normalized, scalers

def inverse_normalize(data, scalers, titles):
    datacopy = data.copy()
    for i in range(len(titles)):
        title = titles[i]
        temp = datacopy[title].values
        temp = temp.reshape(len(temp),1)
        temp = scalers[i].inverse_transform(temp)
        temp = temp.reshape((len(temp)))
        datacopy[title] = temp
    return datacopy

class Dataset(torch.utils.data.Dataset):
    def __init__(self, groups, grp_by, split, features, target, horizon_size):
        self.groups = groups
        self.grp_by = grp_by
        self.split = split
        self.features = features
        self.target = target
        self.horizon_size = horizon_size

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        group = self.groups[idx]

        df = self.grp_by.get_group(group)

        src, trg = split_df(df, split=self.split, horizon_size=self.horizon_size)

        src = src[self.features + self.target]

        src = df_to_np(src)

        trg_in = trg[self.features + [f"{mytarget}_lag_1" for mytarget in self.target]]

        trg_in = np.array(trg_in)
        trg_out = np.array(trg[self.target])

        src = torch.tensor(src, dtype=torch.float)
        trg_in = torch.tensor(trg_in, dtype=torch.float)
        trg_out = torch.tensor(trg_out, dtype=torch.float)

        return src, trg_in, trg_out


def train(
    data_csv_path: str,
    feature_target_names_path: str,
    output_json_path: str,
    log_dir: str = "ts_logs",
    model_dir: str = "ts_models",
    val_csv_path: str = None ,
    batch_size: int = 32,
    epochs: int = 2,
    horizon_size: int = 96,
    
):
    # data = pd.read_csv(data_csv_path)

    with open(feature_target_names_path) as f:
        feature_target_names = json.load(f)

    target_titles = feature_target_names["targets"]
    target_titles_lag = feature_target_names["lag_features"]
    train_data = pd.read_csv(data_csv_path)
    val_data = pd.read_csv(val_csv_path)
  
    train_data, train_scalers = normalize_data(train_data, target_titles)
    train_data, train_scalers_lag = normalize_data(train_data, target_titles_lag)
    val_data, val_scalers = normalize_data(val_data, target_titles)
    val_data, val_scalers_lag = normalize_data(val_data, target_titles_lag)

    train_groups_by_train, train_groups = get_groups(train_data, 96+horizon_size)
    val_groups_by_train , val_groups = get_groups(val_data, 96+horizon_size)
    # print(train_groups_by_train.get_group(1))

    

    train_data = Dataset(
        groups=train_groups,
        grp_by=train_groups_by_train,
        split="train",
        features=feature_target_names["features"],
        target=feature_target_names["targets"],
        horizon_size=horizon_size,
    )
    val_data = Dataset(
        groups=val_groups,
        grp_by=val_groups_by_train,
        split="val",
        features=feature_target_names["features"],
        target=feature_target_names["targets"],
        horizon_size=horizon_size,
    )

    print("len(train_data)", len(train_data))
    print("len(val_data)", len(val_data))

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        num_workers=10,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        num_workers=10,
        shuffle=False,
    )

    model = TimeSeriesForcasting(
        n_encoder_inputs=len(feature_target_names["features"]) + len(feature_target_names["targets"]),
        n_decoder_inputs=len(feature_target_names["features"]) + len(feature_target_names["targets"]),
        lr=1e-5,
        dropout=0.1,
    )

    logger = TensorBoardLogger(
        save_dir=log_dir,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_loss",
        mode="min",
        dirpath=model_dir,
        filename="ts",
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        gpus=1,
        logger=logger,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model, train_loader, val_loader)

    result_val = trainer.test(test_dataloaders=val_loader)

    output_json = {
        "val_loss": result_val[0]["test_loss"],
        "best_model_path": checkpoint_callback.best_model_path,
    }

    if output_json_path is not None:
        with open(output_json_path, "w") as f:
            json.dump(output_json, f, indent=4)

    return output_json


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv_path")
    parser.add_argument("--feature_target_names_path")
    parser.add_argument("--output_json_path", default=None)
    parser.add_argument("--log_dir")
    parser.add_argument("--model_dir")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--val_csv_path")
    args = parser.parse_args()

    train(
        data_csv_path=args.data_csv_path,
        feature_target_names_path=args.feature_target_names_path,
        output_json_path=args.output_json_path,
        log_dir=args.log_dir,
        model_dir=args.model_dir,
        epochs=args.epochs,
        val_csv_path=args.val_csv_path,
    )
