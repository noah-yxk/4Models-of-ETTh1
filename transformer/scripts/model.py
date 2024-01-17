import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import Linear


def smape_loss(y_pred, target):
    loss = 2 * (y_pred - target).abs() / (y_pred.abs() + target.abs() + 1e-8)

    return loss.mean()

def mse_loss(y_pred, target):
    n = len(y_pred)
    loss = sum((y_pred[i] - target[i])**2 for i in range(n)) / n
    return loss


def gen_trg_mask(length, device):
    mask = torch.tril(torch.ones(length, length, device=device)) == 1

    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )

    return mask


class TimeSeriesForcasting(pl.LightningModule):
    def __init__(
        self,
        n_encoder_inputs,
        n_decoder_inputs,
        channels=512,
        dropout=0.1,
        lr=1e-4,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.dropout = dropout

        self.input_pos_embedding = torch.nn.Embedding(1024, embedding_dim=channels)
        self.target_pos_embedding = torch.nn.Embedding(1024, embedding_dim=channels)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=8,
            dropout=self.dropout,
            dim_feedforward=4 * channels,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=channels,
            nhead=8,
            dropout=self.dropout,
            dim_feedforward=4 * channels,
        )

        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=8)
        self.decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=8)

        self.input_projection = Linear(n_encoder_inputs, channels)
        self.output_projection = Linear(n_decoder_inputs, channels)

        self.linear = Linear(channels, 7)

        self.do = nn.Dropout(p=self.dropout)

    def encode_src(self, src):
        src_start = self.input_projection(src).permute(1, 0, 2)

        in_sequence_len, batch_size = src_start.size(0), src_start.size(1)
        pos_encoder = (
            torch.arange(0, in_sequence_len, device=src.device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )

        pos_encoder = self.input_pos_embedding(pos_encoder).permute(1, 0, 2)

        src = src_start + pos_encoder

        src = self.encoder(src) + src_start

        return src

    def decode_trg(self, trg, memory):

        trg_start = self.output_projection(trg).permute(1, 0, 2)

        out_sequence_len, batch_size = trg_start.size(0), trg_start.size(1)

        pos_decoder = (
            torch.arange(0, out_sequence_len, device=trg.device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )
        pos_decoder = self.target_pos_embedding(pos_decoder).permute(1, 0, 2)

        trg = pos_decoder + trg_start

        trg_mask = gen_trg_mask(out_sequence_len, trg.device)

        out = self.decoder(tgt=trg, memory=memory, tgt_mask=trg_mask) + trg_start

        out = out.permute(1, 0, 2)

        out = self.linear(out)

        return out

    def forward(self, x):
        src, trg = x

        src = self.encode_src(src)

        out = self.decode_trg(trg=trg, memory=src)

        return out

    def training_step(self, batch, batch_idx):
        src, trg_in, trg_out = batch

        y_hat = self((src, trg_in)) 
        # y_hat = y_hat[:, :, -1]
        # trg_out = trg_out[:, :, -1]
        # sliced_y_hat_list = [y_hat[:, :, i] for i in range(7)]
        # flat_y_hats = [hat.view(-1) for hat in sliced_y_hat_list]
        y_hat = y_hat.view(-1)
        # sliced_y_list = [trg_out[:, :, i] for i in range(7)]
        # flat_ys = [flat.view(-1) for flat in sliced_y_list]
        y = trg_out.view(-1)
        loss = smape_loss(y_hat, y)

        # losslst = [smape_loss(flat_y_hat, lat_y) for flat_y_hat,lat_y in zip(flat_y_hats,flat_ys)]

        # losslst[6] = losslst[6]
        # loss = sum(losslst)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        src, trg_in, trg_out = batch

        y_hat = self((src, trg_in)) 
        # y_hat = y_hat[:, :, -1]
        # trg_out = trg_out[:, :, -1]
        # sliced_y_hat_list = [y_hat[:, :, i] for i in range(7)]
        # flat_y_hats = [hat.view(-1) for hat in sliced_y_hat_list]
        y_hat = y_hat.view(-1)
        # sliced_y_list = [trg_out[:, :, i] for i in range(7)]
        # flat_ys = [flat.view(-1) for flat in sliced_y_list]
        y = trg_out.view(-1)
        loss = smape_loss(y_hat, y)

        # losslst = [smape_loss(flat_y_hat, lat_y) for flat_y_hat,lat_y in zip(flat_y_hats,flat_ys)]

        # losslst[6] = losslst[6]
        # loss = sum(losslst)
        print("valid_loss", loss)
        self.log("valid_loss", loss)

        return loss

    def test_step(self, batch, batch_idx):
        src, trg_in, trg_out = batch

        y_hat = self((src, trg_in))
        # y_hat = y_hat[:, :, -1]
        # trg_out = trg_out[:, :, -1]

        y_hat = y_hat.view(-1)
        y = trg_out.view(-1)

        loss = smape_loss(y_hat, y)

        self.log("test_loss", loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.1
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "valid_loss",
        }



