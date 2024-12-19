import torch
import numpy as np
from torch import nn
from scipy.ndimage import gaussian_filter

from utils import read_3d


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.ReLU()):
        super().__init__()
        expansion_ratio = 4

        self.encoder_block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=(7, 7),
                stride=1,
                padding=3,
                groups=in_channels,
            ),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(
                in_channels,
                expansion_ratio * out_channels,
                kernel_size=(1, 1),
                stride=1,
            ),
            activation,
            nn.Conv2d(
                expansion_ratio * out_channels,
                out_channels,
                kernel_size=(1, 1),
                stride=1,
            ),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(7, 7),
                stride=1,
                padding=3,
                groups=out_channels,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(
                out_channels,
                expansion_ratio * out_channels,
                kernel_size=(1, 1),
                stride=1,
            ),
            activation,
            nn.Conv2d(
                expansion_ratio * out_channels,
                out_channels,
                kernel_size=(1, 1),
                stride=1,
            ),
        )

    def forward(self, x):
        return self.encoder_block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.ReLU()):
        super().__init__()
        expansion_ratio = 4

        self.decoder_block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=(7, 7),
                stride=1,
                padding=3,
                groups=in_channels,
            ),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(
                in_channels, expansion_ratio * in_channels, kernel_size=(1, 1), stride=1
            ),
            activation,
            nn.Conv2d(
                expansion_ratio * in_channels,
                out_channels,
                kernel_size=(1, 1),
                stride=1,
            ),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(7, 7),
                stride=1,
                padding=3,
                groups=out_channels,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(
                out_channels,
                expansion_ratio * out_channels,
                kernel_size=(1, 1),
                stride=1,
            ),
            activation,
            nn.Conv2d(
                expansion_ratio * out_channels,
                out_channels,
                kernel_size=(1, 1),
                stride=1,
            ),
        )

    def forward(self, x):
        return self.decoder_block(x)


class AttentionResBlock(nn.Module):
    def __init__(self, in_channels, activation=nn.ReLU()):
        super().__init__()
        self.query_conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=(1, 1), stride=1
        )
        self.key_conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=(1, 1), stride=2
        )
        self.attention_conv = nn.Conv2d(in_channels, 1, kernel_size=(1, 1), stride=1)

        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.activation = activation

    def forward(self, query, key, value):
        query = self.query_conv(query)
        key = self.key_conv(key)

        combined_attention = self.activation(query + key)
        attention_map = torch.sigmoid(self.attention_conv(combined_attention))
        upsampled_attention_map = self.upsample(attention_map)
        attention_scores = value * upsampled_attention_map
        return attention_scores


class AttentionUNet(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        # Config
        out_channels = 1  # Mask has 1 channel
        n_filters = 64
        activation = nn.ReLU()

        # Up and downsampling methods
        self.downsample = nn.MaxPool2d((2, 2), stride=2)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        # Encoder
        self.enc_block_1 = EncoderBlock(in_channels, 1 * n_filters, activation)
        self.enc_block_2 = EncoderBlock(1 * n_filters, 2 * n_filters, activation)
        self.enc_block_3 = EncoderBlock(2 * n_filters, 4 * n_filters, activation)
        self.enc_block_4 = EncoderBlock(4 * n_filters, 8 * n_filters, activation)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                8 * n_filters,
                8 * n_filters,
                kernel_size=(7, 7),
                stride=1,
                padding=3,
                groups=8 * n_filters,
            ),
            nn.BatchNorm2d(8 * n_filters),
            nn.Conv2d(8 * n_filters, 4 * 8 * n_filters, kernel_size=(1, 1), stride=1),
            activation,
            nn.Conv2d(4 * 8 * n_filters, 8 * n_filters, kernel_size=(1, 1), stride=1),
            nn.Conv2d(
                8 * n_filters,
                8 * n_filters,
                kernel_size=(7, 7),
                stride=1,
                padding=3,
                groups=8 * n_filters,
            ),
            nn.BatchNorm2d(8 * n_filters),
            nn.Conv2d(8 * n_filters, 4 * 8 * n_filters, kernel_size=(1, 1), stride=1),
            activation,
            nn.Conv2d(4 * 8 * n_filters, 8 * n_filters, kernel_size=(1, 1), stride=1),
        )

        # Decoder
        self.dec_block_4 = DecoderBlock(8 * n_filters, 4 * n_filters, activation)
        self.dec_block_3 = DecoderBlock(4 * n_filters, 2 * n_filters, activation)
        self.dec_block_2 = DecoderBlock(2 * n_filters, 1 * n_filters, activation)
        self.dec_block_1 = DecoderBlock(1 * n_filters, 1 * n_filters, activation)

        # Output projection
        self.output = nn.Conv2d(
            1 * n_filters, out_channels, kernel_size=(1, 1), stride=1, padding=0
        )

        # Attention res blocks
        self.att_res_block_1 = AttentionResBlock(1 * n_filters)
        self.att_res_block_2 = AttentionResBlock(2 * n_filters)
        self.att_res_block_3 = AttentionResBlock(4 * n_filters)
        self.att_res_block_4 = AttentionResBlock(8 * n_filters)

    def forward(self, x):
        # Encoder
        enc_1 = self.enc_block_1(x)
        x = self.downsample(enc_1)
        enc_2 = self.enc_block_2(x)
        x = self.downsample(enc_2)
        enc_3 = self.enc_block_3(x)
        x = self.downsample(enc_3)
        enc_4 = self.enc_block_4(x)
        x = self.downsample(enc_4)

        # Bottleneck
        dec_4 = self.bottleneck(x)

        # Decoder
        x = self.upsample(dec_4)
        att_4 = self.att_res_block_4(dec_4, enc_4, enc_4)  # QKV
        x = torch.add(x, att_4)  # Add attention masked value rather than concat

        dec_3 = self.dec_block_4(x)
        x = self.upsample(dec_3)
        att_3 = self.att_res_block_3(dec_3, enc_3, enc_3)
        x = torch.add(x, att_3)  # Add attention

        dec_2 = self.dec_block_3(x)
        x = self.upsample(dec_2)
        att_2 = self.att_res_block_2(dec_2, enc_2, enc_2)
        x = torch.add(x, att_2)  # Add attention

        dec_1 = self.dec_block_2(x)
        x = self.upsample(dec_1)
        att_1 = self.att_res_block_1(dec_1, enc_1, enc_1)
        x = torch.add(x, att_1)  # Add attention

        x = self.dec_block_1(x)
        x = self.output(x)
        return x


def get_model(flair_present: bool = False) -> AttentionUNet:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AttentionUNet(in_channels=3 if flair_present else 2)
    weights_path = "weights_flair.pth" if flair_present else "weights.pth"
    model.load_state_dict(torch.load(weights_path, map_location=device))
    return model.to(device)


def gaussian_blur(image, sigma):
    sigmas = [0 if i == 0 else sigma for i in range(4)]
    return gaussian_filter(image, sigmas)


def prediction_for_volume(patient_id: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image = read_3d(patient_id)
    flair_present = image.shape[0] == 3
    model = get_model(flair_present)
    predictions = np.zeros(image.shape[1:])
    z_size = image.shape[1]
    for i in range(z_size):
        image_2d = image[:, i, :, :]
        image_min, image_max = np.min(image_2d), np.max(image_2d)
        if image_min != image_max:
            image_2d = (image_2d - image_min) / (image_max - image_min)
        image_2d = torch.tensor(image_2d, dtype=torch.float32).to(device)
        logit = model(image_2d.reshape((1, *image_2d.shape)))
        predictions[i, :, :] = logit.detach().cpu().numpy().squeeze(0).reshape(240, 240)
    gaussian_blur(image, 1.1)
    for i in range(z_size):
        image_2d = predictions[i, :, :]
        image_2d = torch.tensor(image_2d, dtype=torch.float32).to(device)
        pred = torch.sigmoid(image_2d)
        predictions[i, :, :] = pred.detach().cpu().numpy()
    return predictions
