import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset
from tqdm import tqdm
import torch.nn.functional as F
import time
device = torch.device("cuda")

class LSTMEncoder(nn.Module):
    def __init__(self, rnn_num_layers=1, input_feature_len=1, sequence_len=168, hidden_size=100, bidirectional=False):
        super().__init__()
        self.sequence_len = sequence_len
        self.hidden_size = hidden_size
        self.input_feature_len = input_feature_len
        self.num_layers = rnn_num_layers
        self.rnn_directions = 2 if bidirectional else 1
        self.lstm = nn.LSTM(
            num_layers=rnn_num_layers,
            input_size=input_feature_len,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=bidirectional
        )
 
    def forward(self, input_seq):
 
        ht = torch.zeros(self.num_layers * self.rnn_directions, input_seq.size(0), self.hidden_size, device='cuda')
        ct = ht.clone()
        if input_seq.ndim < 3:
            input_seq.unsqueeze_(2)
        lstm_out, (ht, ct) = self.lstm(input_seq, (ht,ct))
        if self.rnn_directions > 1:
            lstm_out = lstm_out.view(input_seq.size(0), self.sequence_len, self.rnn_directions, self.hidden_size)
            lstm_out = torch.sum(lstm_out, axis=2)
        return lstm_out, ht.squeeze(0)
 
class AttentionDecoderCell(nn.Module):
    def __init__(self, input_feature_len, out_put, sequence_len, hidden_size):
        super().__init__()
        # attention - inputs - (decoder_inputs, prev_hidden)
        self.attention_linear = nn.Linear(hidden_size + input_feature_len, sequence_len)
        # attention_combine - inputs - (decoder_inputs, attention * encoder_outputs)
        self.decoder_rnn_cell = nn.LSTMCell(
            input_size=hidden_size,
            hidden_size=hidden_size,
        )
        self.out = nn.Linear(hidden_size, input_feature_len)
 
    def forward(self, encoder_output, prev_hidden, y):
        if prev_hidden.ndimension() == 3:
            prev_hidden = prev_hidden[-1]
        attention_input = torch.cat((prev_hidden, y), axis=1)
        attention_weights = F.softmax(self.attention_linear(attention_input), dim=-1).unsqueeze(1)
        attention_combine = torch.bmm(attention_weights, encoder_output).squeeze(1)
        rnn_hidden, rnn_hidden = self.decoder_rnn_cell(attention_combine, (prev_hidden, prev_hidden))
        output = self.out(rnn_hidden)
        return output, rnn_hidden
 
 
class EncoderDecoderWrapper(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, pred_size, window_size, teacher_forcing=0.3):
        super().__init__()
        self.encoder = LSTMEncoder(num_layers, input_size, window_size, hidden_size)
        self.decoder_cell = AttentionDecoderCell(input_size, output_size,  window_size, hidden_size)
        self.output_size = output_size
        self.input_size = input_size
        self.pred_len = pred_size
        self.teacher_forcing = teacher_forcing
        self.linear = nn.Linear(input_size,output_size)
 
 
    def __call__(self, xb, yb=None):
        input_seq = xb
        encoder_output, encoder_hidden = self.encoder(input_seq)
        prev_hidden = encoder_hidden
        outputs = torch.zeros(self.pred_len, input_seq.size(0), self.input_size, device='cuda')
        y_prev = input_seq[:, -1, :]
        for i in range(self.pred_len):
            if (yb is not None) and (i > 0) and (torch.rand(1) < self.teacher_forcing):
                y_prev = yb[:, i].unsqueeze(1)
            rnn_output, prev_hidden = self.decoder_cell(encoder_output, prev_hidden, y_prev)
            y_prev = rnn_output
            outputs[i, :, :] = rnn_output
        outputs = outputs.permute(1, 0, 2)
        if self.output_size == 1:
            outputs = self.linear(outputs)
        return outputs
    
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
    plt.savefig(f"OT_{len(pred)}_{time.asctime()}.png")
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
    plt.savefig(f"loss_{pred_size}_{time.asctime()}.png")
    plt.close()

 

if __name__ == '__main__':
    device = torch.device("cuda")
    batch_size = 1024
    pred_size = 96
    epochs = 30

    data = get_data(96, pred_size)
    train_loader = get_loader(data["train"][0], data["train"][1], batch_size)
    val_loader = get_loader(data["val"][0], data["val"][1], batch_size)
    test_loader = get_loader(data["test"][0], data["test"][1], batch_size)

    model = EncoderDecoderWrapper(input_size=13, output_size=7, hidden_size=256, num_layers=2, pred_size=pred_size, window_size=96)
    model.to(device)
    loss_function = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    min = float("inf")
    best_model = None
    train_losses = []
    val_losses = []
    for i in tqdm(range(epochs)):
        count = 0
        train_loss = 0
        model.train()
        for seq, labels in train_loader:
            optimizer.zero_grad()
            y_pred = model(seq)
            single_loss = loss_function(y_pred[:, :, 6], labels[:, :, 6])
            single_loss.backward()
            optimizer.step()
            train_loss += single_loss.item()
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
                loss = loss_function(preds[:, :, 6], y[:, :, 6])
                val_loss += loss.item()
                count += 1
        val_loss /= count
        val_losses.append(val_loss)
        print(f"epoch {i+1}: train_loss: {train_loss:.4f}, val_loss:{val_loss:.4f}")
        if val_loss < min:
            best_model = model

    draw_loss(train_losses[5:], val_losses[5:])
    torch.save(best_model.state_dict(), f"best_model_{pred_size}.pt")
    # 在测试集上检验效果
    best_model = model
    best_model.eval()
    test_loss = 0
    count = 0
    with torch.no_grad():
        for x_test, y_test in test_loader:
            y_test_pred = best_model(x_test)
            loss = loss_function(y_test_pred[:, :, 6], y_test[:, :, 6])
            test_loss += loss.item()
            count += 1
    test_loss /= count
    print(f"测试集上的Metric: {test_loss:.4f}")

    test_loader = get_loader(data["test"][0], data["test"][1], batch_size)
    best_model.eval()
    history = None
    groundTruth = None
    pred = None
    min = float("inf")
    with torch.no_grad():
        for data in test_loader:
            x_test, y_test = data
            y_test_pred = best_model(x_test)
            loss = loss_function(y_test_pred[:, :, 6], y_test[:, :, 6])
            if loss.item() < min:
                pred = y_test_pred
                history = x_test
                groundTruth = y_test
                min = loss.item()
                break

    draw_OT(history[0, :, :].cpu().numpy(), groundTruth[0, :, :].cpu().numpy(), y_test_pred[0, :, :].cpu().numpy(), 6)

