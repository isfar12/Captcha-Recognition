import torch
import torch.nn as nn
import torch.nn.functional as F


class CaptchaModel(nn.Module):
    def __init__(self, num_characters):
        super(CaptchaModel, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=128, kernel_size=3, padding=1)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(
            in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 64 channels, height 10 after conv and pooling
        self.linear = nn.Linear(64*10, 64)
        self.dropout = nn.Dropout(0.3)
        self.gru = nn.GRU(64, 32, bidirectional=True,
                          num_layers=2, batch_first=True, dropout=0.3,)
        self.classifier = nn.Linear(
            32*2, num_characters+1)  # +1 for blank/padding

    def forward(self, images, targets=None):
        batch_size, channels, height, width = images.size()
        # print(batch_size, channels, height, width)
        x = F.relu(self.conv1(images))
        # print(x.size())
        x = self.max_pool1(x)
        # print(x.size())
        x = F.relu(self.conv2(x))
        # print(x.size())  # batch, channels, height, width
        x = self.max_pool2(x)  # 2, 64, 10, 37
        # print(x.size())
        # batch, width, channels, height -> 2, 37, 64, 10
        x = x.permute(0, 3, 1, 2)
        # we use permute to bring width to second dimension so that RNN can process it as sequence
        # print(x.size())
        # batch, width, channels*height -> 2, 37, 640 we flatten channels and height
        x = x.view(batch_size, x.size(1), -1)
        # print(x.size())
        # Now, x is ready to be fed into RNN layers for sequence modeling
        x = self.linear(x)  # batch, width, num_characters
        x = self.dropout(x)
        # print(x.size())
        x, _ = self.gru(x)  # batch, width, hidden_size*2
        # print(x.size())
        x = self.classifier(x)  # batch, width, num_characters+1
        # print(x.size())
        x = x.permute(1, 0, 2)  # for CTC Loss: width, batch, num_characters+1
        # print(x.size())
        if targets is not None:
            log_softmax = F.log_softmax(x, dim=2)
            input_lengths = torch.full(
                size=(batch_size,),
                fill_value=x.size(0),
                dtype=torch.long
            )
            # print(input_lengths)
            target_lengths = torch.full(
                size=(batch_size,),
                fill_value=targets.size(1),
                dtype=torch.long
            )
            # print(target_lengths)
            ctc_loss = nn.CTCLoss(blank=0)
            loss = ctc_loss(
                log_softmax,
                targets,
                input_lengths,
                target_lengths
            )
            return x, loss

        return x, None


if __name__ == "__main__":
    model = CaptchaModel(num_characters=36)
    sample_input = torch.randn(2, 3, 40, 150)
    outputs, loss = model(sample_input, targets=torch.randint(1, 37, (2, 5)))
"""
2 3 40 150
torch.Size([2, 128, 40, 150])
torch.Size([2, 128, 20, 75])
torch.Size([2, 64, 20, 75])
torch.Size([2, 64, 10, 37])
torch.Size([2, 37, 64, 10])
torch.Size([2, 37, 640])
torch.Size([2, 37, 64])
torch.Size([2, 37, 64])
torch.Size([2, 37, 37])
torch.Size([37, 2, 37])
tensor([37, 37])
tensor([5, 5])
"""
