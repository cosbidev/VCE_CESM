import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
       # CON33 RELU BN RESIDUALS ADD CUSTOM
        self.enc1_1 = nn.Sequential(
            nn.Conv2d(1, 24, 3, padding=1),
            # 1 = input channel, 24 = output channels , 3 = kernel size
            nn.BatchNorm2d(24),
            nn.Conv2d(24, 24, 3, padding=1),
            nn.BatchNorm2d(24)
        )

        self.enc1_2 = nn.Sequential(
            nn.Conv2d(1, 24, 3, padding=1),
            nn.BatchNorm2d(24)
        )

        self.enc1_3 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(24)
        )

        self.enc2_1 = nn.Sequential(
            nn.Conv2d(24, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.Conv2d(48, 48, 3, padding=1),
            nn.BatchNorm2d(48)
        )

        self.enc2_2 = nn.Sequential(
            nn.Conv2d(24, 48, 3, padding=1),
            nn.BatchNorm2d(48)
        )

        self.enc2_3 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(48)
        )

        self.enc3_1 = nn.Sequential(
            nn.Conv2d(48, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.Conv2d(96, 96, 3, padding=1),
            nn.BatchNorm2d(96)
        )

        self.enc3_2 = nn.Sequential(
            nn.Conv2d(48, 96, 3, padding=1),
            nn.BatchNorm2d(96)
        )

        self.enc3_3 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(96)
        )

        self.enc4_1 = nn.Sequential(
            nn.Conv2d(96, 192, 3, padding=1),
            nn.BatchNorm2d(192),
            nn.Conv2d(192, 192, 3, padding=1),
            nn.BatchNorm2d(192)
        )

        self.enc4_2 = nn.Sequential(
            nn.Conv2d(96, 192, 3, padding=1),
            nn.BatchNorm2d(192)
        )

        self.enc4_3 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(192)
        )

        self.dec1_1 = nn.Sequential(
            nn.Conv2d(288, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.Conv2d(96, 96, 3, padding=1),
            nn.BatchNorm2d(96)
        )

        self.dec1_2 = nn.Sequential(
            nn.Conv2d(288, 96, 3, padding=1),
            nn.BatchNorm2d(96)
        )

        self.dec1_3 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(96)
        )

        self.dec2_1 = nn.Sequential(
            nn.Conv2d(144, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.Conv2d(48, 48, 3, padding=1),
            nn.BatchNorm2d(48)
        )

        self.dec2_2 = nn.Sequential(
            nn.Conv2d(144, 48, 3, padding=1),
            nn.BatchNorm2d(48)
        )

        self.dec2_3 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(48)
        )

        self.dec3_1 = nn.Sequential(
            nn.Conv2d(72, 24, 3, padding=1),
            nn.BatchNorm2d(24),
            nn.Conv2d(24, 24, 3, padding=1),
            nn.BatchNorm2d(24)
        )

        self.dec3_2 = nn.Sequential(
            nn.Conv2d(72, 24, 3, padding=1),
            nn.BatchNorm2d(24)
        )

        self.dec3_3 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(24)
        )

        self.dec4 = nn.Sequential(
            nn.Conv2d(24, 1, 1),
            nn.Sigmoid()
        )

        self.MP = nn.Sequential(
           nn.MaxPool2d(2, stride=2)
        )

        self.US = nn.Sequential(
            nn.Upsample(scale_factor=2)
        )


    def forward(self, x):
        x1_1 = self.enc1_1(x)
        x1_2 = self.enc1_2(x)
        enc_1 = self.enc1_3(x1_1 + x1_2)
        enc_1_mp = self.MP(enc_1)

        x2_1 = self.enc2_1(enc_1_mp)
        x2_2 = self.enc2_2(enc_1_mp)
        enc_2 = self.enc2_3(x2_1 + x2_2)
        enc_2_mp = self.MP(enc_2)

        x3_1 = self.enc3_1(enc_2_mp)
        x3_2 = self.enc3_2(enc_2_mp)
        enc_3 = self.enc3_3(x3_1 + x3_2)
        enc_3_mp = self.MP(enc_3)

        x4_1 = self.enc4_1(enc_3_mp)
        x4_2 = self.enc4_2(enc_3_mp)
        enc_4 = self.enc4_3(x4_1 + x4_2)
        enc_4_mp = self.US(enc_4)

        conc_1 = torch.cat((enc_3, enc_4_mp), dim=1)

        y1_1 = self.dec1_1(conc_1)
        y1_2 = self.dec1_2(conc_1)
        dec_1 = self.dec1_3(y1_1 + y1_2)
        dec_1_us = self.US(dec_1)

        conc_2 = torch.cat((enc_2, dec_1_us), dim=1)

        y2_1 = self.dec2_1(conc_2)
        y2_2 = self.dec2_2(conc_2)
        dec_2 = self.dec2_3(y2_1 + y2_2)
        dec_2_us = self.US(dec_2)

        conc_3 = torch.cat((enc_1, dec_2_us), dim=1)

        y3_1 = self.dec3_1(conc_3)
        y3_2 = self.dec3_2(conc_3)
        dec_3 = self.dec3_3(y3_1 + y3_2)

        output = self.dec4(dec_3)

        return output



