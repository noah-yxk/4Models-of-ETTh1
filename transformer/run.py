from time_series_forecasting import training, evaluation, plot_images
import os
import random
import numpy as np
import torch
import pandas as pd

current_time = pd.Timestamp.now().strftime("%Y-%m-%d-%H-%M-%S")

epochs = 50
horizon_size = 96
data_csv_path = "data/processed_train_data.csv"
val_csv_path = "data/processed_val_data.csv"
feature_target_names_path = "data/config.json"

log_dir = f"models/ts_views_logs_{horizon_size}_{current_time}"
model_dir = f"models/ts_views_models_{horizon_size}_{current_time}"



def set_seed(seed, cuda=False):
    '''
    设置随机数种子的函数。通过调用该函数，可以设置numpy、random和torch的随机数种子，以实现结果的可重复性。
    Parameters
    ----------
    seed 设置的随机数种子的值
    cuda 表示是否使用CUDA加速

    Returns
    -------
    '''
    # 分别设置了numpy和Python内置的random模块的随机数种子
    np.random.seed(seed)
    random.seed(seed)
    # 设置PyTorch库的非CUDA和CUDA的随机数种子
    # 如果为True，则会调用torch.cuda.manual_seed(seed)设置CUDA相关的随机数种子；
    # 如果为False，则调用torch.manual_seed(seed)设置非CUDA相关的随机数种子。
    if cuda:
        torch.cuda.manual_seed(seed)
    else:
        torch.manual_seed(seed)

# for i in range(5,10):
#     set_seed(i+10, cuda=True)
#     output_json_path = f"models/trained_config_{horizon_size}_{i}_{current_time}.json"
#     training.train(
#             data_csv_path=data_csv_path,
#             feature_target_names_path=feature_target_names_path,
#             output_json_path=output_json_path,
#             log_dir=log_dir,
#             model_dir=model_dir,
#             epochs=epochs,
#             val_csv_path=val_csv_path,
#             horizon_size=horizon_size,

#         )


data_csv_path = "data/processed_test_data.csv"
train_start_time = current_time

for i in range(0,5):
    # set_seed(i+10)
    rount_name = f"round_{i}_{horizon_size}_{current_time}"
    os.makedirs(f"data/{rount_name}", exist_ok=True)
    trained_json_path=f"models/trained_config_{horizon_size}_{i}.json"
    eval_json_path=f"data/{rount_name}/eval_{horizon_size}.json"
    data_for_visualization_path=f"data/{rount_name}/visualization_{horizon_size}.json"

    evaluation.evaluate(
        data_csv_path=data_csv_path,
        feature_target_names_path=feature_target_names_path,
        trained_json_path=trained_json_path,
        eval_json_path=eval_json_path,
        data_for_visualization_path=data_for_visualization_path,
        horizon_size=horizon_size,

    )

    plot_images.draw(visualization_path=data_for_visualization_path, dirname=f"data/{rount_name}/images_{horizon_size}",)