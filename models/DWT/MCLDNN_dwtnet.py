import torch
import torch.nn as nn
from models.DWT.WTConv1d import WTConv1d

class MCLDNN(nn.Module):
    def __init__(self, num_classes):
        super(MCLDNN, self).__init__()
        # self.dwt = WTConv1d(in_channels=2, out_channels=2, kernel_size=5, wt_levels=1)
        # Part-A: Multi-channel Inputs and Spatial Characteristics Mapping Section
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=50, kernel_size=(2, 8), padding="same"),  # glorot_uniform
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.ConstantPad1d((7, 0), 0),
            nn.Conv1d(in_channels=1, out_channels=50, kernel_size=8),  # glorot_uniform
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.ConstantPad1d((7, 0), 0),
            nn.Conv1d(in_channels=1, out_channels=50, kernel_size=8),  # glorot_uniform
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=50, out_channels=50, kernel_size=(1, 8), padding="same"),  # glorot_uniform
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=100, out_channels=100, kernel_size=(2, 5)),  # glorot_uniform
            nn.ReLU(),
        )
        # Part-B: TRemporal Characteristics Extraction Section
        self.lstm1 = nn.LSTM(input_size=100, hidden_size=128, num_layers=1,
                            batch_first=True)  # kernel_initializer='glorot_uniform' recurrent_initializer='orthogonal'
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=128, num_layers=1,
                            batch_first=True)
        # Part-C: Fully Connected Classifier
        self.fc1 = nn.Sequential(
            nn.Linear(128, 128),  # glorot_uniform
            nn.SELU(),
            nn.Dropout(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128, 128),  # glorot_uniform
            nn.SELU(),
            nn.Dropout(),
        )
        self.softmax = nn.Sequential(
            nn.Linear(128, num_classes),  # glorot_uniform
            # nn.Softmax(dim=1)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(param)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(param)
                    elif "bias" in name:
                        param.data.zero_()

    def forward(self, batch_x):
        # batch_x = self.dwt(batch_x)
        # Part-A: Multi-channel Inputs and Spatial Characteristics Mapping Section
        conv2_in, conv3_in = batch_x[:, 0:1], batch_x[:, 1:2]
        conv2_out, conv3_out = self.conv2(conv2_in), self.conv3(conv3_in)
        conv2_out, conv3_out = torch.unsqueeze(conv2_out, dim=2), torch.unsqueeze(conv3_out, dim=2)

        concatenate1 = torch.concatenate((conv2_out, conv3_out), dim=2)
        conv4_out = self.conv4(concatenate1)

        batch_x = torch.unsqueeze(batch_x, dim=1)
        conv1_out = self.conv1(batch_x)
        concatenate2 = torch.concatenate((conv4_out, conv1_out), dim=1)
        conv5_out = self.conv5(concatenate2)
        conv5_out = conv5_out.permute(0, 3, 2, 1).flatten(2)
        # Part-B: TRemporal Characteristics Extraction Section
        outputs, _ = self.lstm1(conv5_out)
        outputs, _ = self.lstm2(outputs)
        # Part-C: Fully Connected Classifier
        outputs = self.fc1(outputs[:, -1])
        outputs = self.fc2(outputs)
        outputs = self.softmax(outputs)

        return outputs


if __name__ == '__main__':
    model = MCLDNN(11)
    model = model.cuda()
    input_x = torch.rand(1, 2, 128).cuda()
    # start = time.time()
    output = model(input_x)
    # end = time.time()
    # print((end-start)/400.0)
    # print(model)

    ### thop cal ###
    macs, params = profile(model, inputs=(input_x,))
    print(f"FLOPS: {macs / 1e6:.2f}M")
    print(f"params: {params / 1e3:.2f}K")