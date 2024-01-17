import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import TensorDataset
from tqdm import tqdm

class Seq2Seq_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, pred_size, num_layers) -> None:
        super().__init__()
        self.pred_size = pred_size
        self.hidden_size = hidden_size
        self.embedding = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, hidden_size),
        )
        self.encoder = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers)
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, output_size),
        )

    def forward(self, input_seq):
        input_seq = self.embedding(input_seq.permute(1, 0, 2))
        _, (h, c) = self.encoder(input_seq)
        lstm_out = torch.zeros((1, input_seq.shape[1], input_seq.shape[2]), device=device)
        outputs = []
        for i in range(self.pred_size):
            lstm_out, (h, c) = self.decoder(lstm_out, (h, c))
            outputs.append(lstm_out)          
        return self.fc(torch.cat(outputs)).permute(1, 0, 2)

def split_data(data, x_size, y_size):
    dataX = []  # 保存X
    dataY = []  # 保存Y

    for index in range(len(data)):
        if index + x_size + y_size >= len(data):
            break
        dataX.append(data[index : index + x_size])
        dataY.append(data[index + x_size : index + x_size + y_size])
    dataX = np.array(dataX)
    dataY = np.array(dataY)

    return [dataX, dataY]

def add_date_cols(df, date_col):
    date_col = pd.to_datetime(date_col, format="%Y-%m-%d %H:%M")
    df["hour_of_day"] = date_col.dt.hour / 24
    df["day_of_month"] = date_col.dt.day / 31
    df["day_of_year"] = date_col.dt.dayofyear / 365
    df["month"] =date_col.dt.month / 12
    df["week_of_year"] = date_col.dt.isocalendar().week / 53
    df["year"] = (date_col.dt.year - 2015) / 2
    return df

def get_data(x_size, y_size):
    paths = {"train" : "../train_set.csv", "val" : "../validation_set.csv", "test" : "../test_set.csv"}
    res = {}
    for k, v in paths.items():
        _ = pd.read_csv(v)
        df, date_col = _.iloc[:, 1:], _.iloc[:, 0]
        df = (df - df.min()) / (df.max() - df.min())
        data = add_date_cols(df, date_col)
        x, y = split_data(data.to_numpy(), x_size, y_size)
        res[k] = [x, y]
    return res

def get_loader(x, y, batch_size):

    x_tensor = torch.from_numpy(x.astype(float)).to(torch.float32).to(device)
    y_tensor = torch.from_numpy(y.astype(float)).to(torch.float32).to(device)

    data = TensorDataset(x_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(data, batch_size, drop_last=True)
    return loader

def draw_OT(history, groundTruth, pred, k):
    
    # 绘制每个特征随时间变化的折线图
    plt.figure()
    plt.plot(range(len(history)), history[:, k], label="History")
    plt.plot(range(len(history), len(history) + len(groundTruth)), groundTruth[:, k], label="Ground Truth")
    plt.plot(range(len(history), len(history) + len(groundTruth)), pred[:, k], label="Prediction")
    plt.plot(range(len(history)-1, len(history) + 1), [history[-1, k], groundTruth[0, k]])
    # 添加标签和图例
    plt.xlabel('Time Steps')
    plt.ylabel('Feature Values')
    plt.title('Time Series Data Analysis')
    plt.legend()            
    plt.savefig(f"OT_{len(pred)}.png")
    plt.close()

def draw_loss(train_losses, val_losses):
    plt.figure()
    plt.plot(train_losses, label="train_loss")
    plt.plot(val_losses, label="val_loss")
    # 添加标签和图例
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('loss over epochs')
    plt.legend()            
    plt.savefig(f"loss_{pred_size}.png")
    plt.close()


def create_lr_scheduler(optimizer,
                        num_step: int,  # every epoch has how much step
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,  # warmup进行多少个epoch
                        warmup_factor=1e-3):
    """
    :param optimizer: 优化器
    :param num_step: 每个epoch迭代多少次，len(data_loader)
    :param epochs: 总共训练多少个epoch
    :param warmup: 是否采用warmup
    :param warmup_epochs: warmup进行多少个epoch
    :param warmup_factor: warmup的一个倍数因子
    :return:
    """
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0
    def f(x):
        """
        :x epoch或者iteration
        :return 根据step数返回一个学习率倍率因子
        注意在训练开始之前，pytorch似乎会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子大小从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha  # 对于alpha的一个线性变换，alpha是关于x的一个反比例函数变化
        else:
            # warmup后lr的倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9
        # （1-a/b）^0.9 b是当前这个epoch结束训练总共了多少次了（除去warmup），这个关系是指一个epcoch中
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)

if __name__ == "__main__":
    device = torch.device("cuda")
    batch_size = 1024
    pred_size = 336
    epochs = 50

    data = get_data(96, pred_size)
    train_loader = get_loader(data["train"][0], data["train"][1], batch_size)
    val_loader = get_loader(data["val"][0], data["val"][1], batch_size)
    test_loader = get_loader(data["test"][0], data["test"][1], batch_size)

    model = Seq2Seq_LSTM(input_size=13, hidden_size=256, output_size=7, pred_size=pred_size, num_layers=2)
    model.to(device)
    loss_function = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=1e-3)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), epochs, warmup=True, warmup_epochs=5)

    min = float("inf")
    best_model = None
    train_losses = []
    val_losses = []
    for epoch in range(1, epochs + 1):
        count = 0
        train_loss = 0
        model.train()
        train_bar = tqdm(train_loader, leave=False)
        if epoch > 15 and epoch % 10 == 0:      
            for params in optimizer.param_groups:             
                # 遍历Optimizer中的每一组参数，将该组参数的学习率 * 0.9            
                params['lr'] *= 0.1

        for x, y in train_bar:
            optimizer.zero_grad()
            preds = model(x)
            loss = loss_function(preds[:, :, 6], y[:, :, 6])
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            train_bar.desc = "train epoch[{}/{}] loss:{:.4f}".format(epoch, epochs, loss.item())
            train_loss += loss.item()
            count += 1
        train_loss /= count
        train_losses.append(train_loss)

        # 模型验证, 保留在验证集上效果最好的模型
        model.eval()
        val_loss = 0
        count = 0
        with torch.no_grad():
            for x, y in val_loader:
                preds = model(x)
                # loss = 0
                # for k in range(y_val_pred.shape[-1]):
                #     loss += loss_function(y_val_pred[:, :, k], y_val[:, :, k])
                # loss /= y_val_pred.shape[-1]
                loss = loss_function(preds[:, :, 6], y[:, :, 6])
                val_loss += loss.item()
                count += 1
        val_loss /= count
        val_losses.append(val_loss)
        print(f"epoch {epoch}: train_loss: {train_loss:.4f}, val_loss:{val_loss:.4f}")
        if val_loss < min:
            best_model = model

    torch.save(best_model.state_dict(), f"best_model_{pred_size}.pt")
    draw_loss(train_losses, val_losses)

    # 在测试集上检验效果
    best_model.eval()
    test_loss = 0
    count = 0
    with torch.no_grad():
        for x_test, y_test in test_loader:
            y_test_pred = best_model(x_test)
            # loss = 0
            # for k in range(y_test_pred.shape[-1]):
            #     loss += loss_function(y_test_pred[:, :, k], y_test[:, :, k])
            # loss /= y_test_pred.shape[-1]
            loss = loss_function(y_test_pred[:, :, 6], y_test[:, :, 6])
            test_loss += loss.item()
            count += 1
    test_loss /= count
    print(f"测试集上的Metric: {test_loss:.4f}")

    test_loader = get_loader(data["test"][0], data["test"][1], 1)
    best_model.eval()
    history = None
    groundTruth = None
    pred = None
    min = float("inf")
    with torch.no_grad():
        for data in test_loader:
            x_test, y_test = data
            y_test_pred = best_model(x_test)
            # loss = 0
            # for k in range(y_test_pred.shape[-1]):
            #     loss += loss_function(y_test_pred[:, :, k], y_test[:, :, k])
            # loss /= y_test_pred.shape[-1]
            loss = loss_function(y_test_pred[:, :, 6], y_test[:, :, 6])
            if loss.item() < min:
                pred = y_test_pred
                history = x_test
                groundTruth = y_test
                min = loss.item()
    # for i in range(7):
    #     show(groundTruth[0, :, :].cpu().numpy(), y_test_pred[0, :, :].cpu().numpy(), i)
    draw_OT(history[0, :, :].cpu().numpy(), groundTruth[0, :, :].cpu().numpy(), y_test_pred[0, :, :].cpu().numpy(), 6)
