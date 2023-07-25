import torch

class EncoderDecoder(torch.nn.Module):
    def __init__(self):
        super(EncoderDecoder, self).__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv1d(3, 6, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(6, 12, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(12, 24, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(24, 48, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU()
        )

        # Decoder layers
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(48, 24, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose1d(24, 12, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose1d(12, 6, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose1d(6, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.ReLU()
        )

        # Skip connections
        self.skip_connections = [
            torch.nn.ConvTranspose1d(24, 24, kernel_size=1, stride=1),
            torch.nn.ConvTranspose1d(12, 12, kernel_size=1, stride=1),
            torch.nn.ConvTranspose1d(6, 6, kernel_size=1, stride=1)
        ]


    def forward(self, x):
        # Encoder pass
        x_1_conv = self.encoder[0](x)
        x_1_conv = self.encoder[1](x_1_conv)
        skip_1 = x_1_conv
        x_2_conv = self.encoder[2](x_1_conv)
        x_2_conv = self.encoder[3](x_2_conv)
        skip_2 = x_2_conv
        x_3_conv = self.encoder[4](x_2_conv)
        x_3_conv = self.encoder[5](x_3_conv)
        skip_3 = x_3_conv
        x_4_conv = self.encoder[6](x_3_conv)
        x_4_conv = self.encoder[7](x_4_conv)
        skip_4 = x_4_conv

        re_x_3_conv = self.decoder[0](x_4_conv)
        re_x_3_conv += skip_3
        re_x_3_conv = self.decoder[1](re_x_3_conv)
        re_x_2_conv = self.decoder[2](re_x_3_conv)
        re_x_2_conv += skip_2
        re_x_2_conv = self.decoder[3](re_x_2_conv)
        re_x_1_conv = self.decoder[4](re_x_2_conv)
        re_x_1_conv += skip_1
        re_x_1_conv = self.decoder[5](re_x_1_conv)
        decoded = self.decoder[6](re_x_1_conv)
        decoded = self.decoder[7](decoded)
        # KEEP THE SAME DIMENSIONS AS INPUT
        if x.shape[2] < decoded.shape[2]:
            # LIMIT TENSOR
            decoded = decoded[:, :, 0:x.shape[2]]
        elif x.shape[2] > decoded.shape[2]:
            # APPEND ADDITIONAL VALUES
            size = x.shape[2] - decoded.shape[2]
            decoded = torch.cat((decoded, x[:, :, -size:]), dim=2)

        return decoded
