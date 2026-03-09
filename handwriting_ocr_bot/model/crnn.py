import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import ALPHABET, BLANK_IDX

NUM_CLASSES = len(ALPHABET) + 1


class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size,
            bidirectional=True,
            batch_first=False,
        )
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out)


class CRNN(nn.Module):
    def __init__(
        self,
        img_height: int = 64,
        lstm_hidden: int = 256,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()

 
        self.cnn = nn.Sequential(
            #Block 1
            nn.Conv2d(1, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), (2, 2)),

            #Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), (2, 2)),

            #Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),

            #Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),

            #Block 5
            nn.Conv2d(512, 512, kernel_size=2, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        )

        cnn_out_h = self._get_cnn_height(img_height)
        cnn_out_channels = 512
        rnn_input_size = cnn_out_channels * cnn_out_h

        self.rnn = nn.Sequential(
            BidirectionalLSTM(rnn_input_size, lstm_hidden, lstm_hidden),
            BidirectionalLSTM(lstm_hidden, lstm_hidden, num_classes),
        )

        last_fc = self.rnn[-1].fc
        nn.init.xavier_uniform_(last_fc.weight)
        nn.init.zeros_(last_fc.bias)
        last_fc.bias.data[0] = -2.0

    def _get_cnn_height(self, img_height: int) -> int:
        x = torch.zeros(1, 1, img_height, 32)
        with torch.no_grad():
            out = self.cnn(x)
        return out.shape[2]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.cnn(x)

        b, c, h, w = feat.shape
        feat = feat.permute(3, 0, 1, 2)
        feat = feat.reshape(w, b, c * h)
        out = self.rnn(feat)

        return F.log_softmax(out, dim=2)


def load_model(checkpoint_path: str, device: torch.device) -> CRNN:
    model = CRNN()
    state = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model