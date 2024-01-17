import json
from typing import Optional
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from time_series_forecasting.model import TimeSeriesForcasting
from time_series_forecasting.training import split_df, Dataset



def mse(true, pred):
    true = np.array(true)
    pred = np.array(pred)
    mse_value = np.mean((pred - true)**2)
    
    return mse_value

def evaluate_regression(true, pred, targets):
    """
    eval mae + mse
    :param true:
    :param pred:
    :return:
    """
    true_lst = []
    pred_lst = []
    for i in range(len(targets)):
        true_lst.append([item[i] for item in true])
        pred_lst.append([item[i] for item in pred])
    return {"mse": [mse(truei, predi) for truei,predi in zip(true_lst,pred_lst)], "mae": [mean_absolute_error(truei, predi) for truei,predi in zip(true_lst,pred_lst)]}

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

def df_to_lst(df, titles):
    lsts = [df[title].tolist() for title in titles]
    return [[lsts[i][j] for i in range(len(titles))] for j in range(len(lsts[0]))]

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

def inverse_normalize_lst(lst: np.array, scaler):


    temp = lst.copy()
    temp = temp.reshape(len(temp),1)
    temp = scaler.inverse_transform(temp)
    temp = temp.reshape((len(temp)))
    
    return temp

def evaluate(
    data_csv_path: str,
    feature_target_names_path: str,
    trained_json_path: str,
    eval_json_path: str,
    horizon_size: int,
    data_for_visualization_path: Optional[str] = None,

):



    with open(trained_json_path) as f:
        model_json = json.load(f)

    model_path = model_json["best_model_path"]
    with open(feature_target_names_path) as f:
        feature_target_names = json.load(f)
    target = feature_target_names["targets"]
    target_titles_lag = feature_target_names["lag_features"]
    train_data = pd.read_csv(data_csv_path)
    train_data, test_scalers = normalize_data(train_data, target)
    train_data, test_scalers_lag = normalize_data(train_data, target_titles_lag)
    train_groups_by_train, train_groups = get_groups(train_data, horizon_size+96)

    val_data = Dataset(
        groups=train_groups,
        grp_by=train_groups_by_train,
        split="val",
        features=feature_target_names["features"],
        target=feature_target_names["targets"],
        horizon_size=horizon_size,
    )

    model = TimeSeriesForcasting(
        n_encoder_inputs=len(feature_target_names["features"]) + len(feature_target_names["targets"]),
        n_decoder_inputs=len(feature_target_names["features"]) + len(feature_target_names["targets"]),
        lr=1e-4,
        dropout=0.5,
    )
    model.load_state_dict(torch.load(model_path)["state_dict"])

    model.eval()

    gt = []
    baseline_last_known_values = []
    neural_predictions = []

    data_for_visualization = []


    
    for i, group in tqdm(enumerate(train_groups)):
        time_series_data = {"history": [], "ground_truth": [], "prediction": []}

        df = train_groups_by_train.get_group(group)
        src, trg = split_df(df, split="val", horizon_size=horizon_size)

        time_series_data["history"] = df_to_lst(src[target], target)[-96:] #df_to_lst(inverse_normalize(src[target],test_scalers,target), target)[-96:]
        time_series_data["ground_truth"] = df_to_lst(trg[target], target)#df_to_lst(inverse_normalize(trg[target],test_scalers,target), target)

        last_known_value = src[target].values[-1]

        trg["last_known_value"] = [last_known_value] * len(trg)

        gt += df_to_lst(trg[target], target)
        baseline_last_known_values += [last_known_value] * len(trg)

        src, trg_in, _ = val_data[i]

        src, trg_in = src.unsqueeze(0), trg_in.unsqueeze(0)

        with torch.no_grad():
            prediction = model((src, trg_in[:, :1, :]))
            for j in range(1, horizon_size):
                last_prediction = prediction[0, -1]
                for k in range(len(last_prediction)):
                    trg_in[:, j, -1-k] = last_prediction[-1-k]
                prediction = model((src, trg_in[:, : (j + 1), :]))

            trg["target_predicted"] = (prediction.squeeze().numpy()).tolist()

            neural_predictions += trg["target_predicted"].tolist()
            
            temp = trg["target_predicted"].tolist()

            time_series_data["prediction"] = temp# temp_array.tolist()


        data_for_visualization.append(time_series_data)






    baseline_eval = evaluate_regression(gt, baseline_last_known_values, target)
    model_eval = evaluate_regression(gt, neural_predictions, target)

    eval_dict = {
        "Baseline_MAE": baseline_eval["mae"],
        "Baseline_MSE": baseline_eval["mse"],
        "Model_MAE": model_eval["mae"],
        "Model_MSE": model_eval["mse"],
    }

    if eval_json_path is not None:
        with open(eval_json_path, "w") as f:
            json.dump(eval_dict, f, indent=4)

    if data_for_visualization_path is not None:
        with open(data_for_visualization_path, "w") as f:
            json.dump(data_for_visualization, f, indent=4)
    

    for k, v in eval_dict.items():
        print(k, v)

    return eval_dict


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv_path")
    parser.add_argument("--feature_target_names_path")
    parser.add_argument("--trained_json_path")
    parser.add_argument("--eval_json_path", default=None)
    parser.add_argument("--data_for_visualization_path", default=None)


    args = parser.parse_args()

    evaluate(
        data_csv_path=args.data_csv_path,
        feature_target_names_path=args.feature_target_names_path,
        trained_json_path=args.trained_json_path,
        eval_json_path=args.eval_json_path,
        data_for_visualization_path=args.data_for_visualization_path,

    )
