import torch

class EncoderDecoder(torch.nn.Module):
    def __init__(self):
        super(EncoderDecoder, self).__init__()

        # Encoder layers
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

    def forward(self, x):
        # Encode the input
        encoded = self.encoder(x)

        # Decode the encoded input
        decoded = self.decoder(encoded)
        
        # KEEP THE SAME DIMENSIONS AS INPUT
        if x.shape[2]<decoded.shape[2]:
            # LIMIT TENSOR
            decoded = decoded[:,:,0:x.shape[2]]
        elif x.shape[2]>decoded.shape[2]:
            # APPEND ADDITIONAL VALUES
            size = x.shape[2]-decoded.shape[2]
            decoded = torch.cat((decoded,x[:,:,-size:]),dim=2)

        return decoded
