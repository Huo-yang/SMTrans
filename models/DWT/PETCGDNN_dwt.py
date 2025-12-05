import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile
from models.DWT.WTConv1d import WTConv1d

'''《An Efficient Deep Learning Model for Automatic Modulation Recognition Based on Parameter Estimation and Transformation》'''


class PETCGDNN(nn.Module):
    def __init__(self, num_classes: int = 11):
        super(PETCGDNN, self).__init__()
        self.dwt = WTConv1d(in_channels=2, out_channels=2, kernel_size=5, wt_levels=1)
        self.phi_fc = nn.Linear(in_features=2 * 128, out_features=1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=75, kernel_size=(2, 8)),  # glorot_uniform
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=75, out_channels=25, kernel_size=(1, 5)),  # glorot_uniform
            nn.ReLU(),
        )
        self.gru = nn.GRU(input_size=25, hidden_size=128,
                          batch_first=True)  # kernel_initializer='glorot_uniform' recurrent_initializer='orthogonal'
        self.classify = nn.Sequential(
            nn.Linear(128, num_classes),  # glorot_uniform
            # nn.Softmax(dim=1)
        )
        # self.apply(self._init_weights)

    def _init_weights(self, m):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(param)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(param)
                    elif "bias" in name:
                        param.data.zero_()

    def forward(self, batch_x: torch.Tensor):
        batch_x = self.dwt(batch_x) + batch_x
        # Part 1
        part1_batch_x = batch_x.flatten(start_dim=1)
        part1_batch_x = self.phi_fc(part1_batch_x)
        phi = self.linear_activation(part1_batch_x)
        # Part 2
        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)
        batch_i = batch_x[:, 0]
        batch_q = batch_x[:, 1]
        transformation_add = batch_i * cos_phi + batch_q * sin_phi
        transformation_minus = batch_q * cos_phi - batch_i * sin_phi
        transformation_add = torch.unsqueeze(transformation_add, dim=1)
        transformation_minus = torch.unsqueeze(transformation_minus, dim=1)
        batch_transformation = torch.concatenate((transformation_add, transformation_minus), dim=1)
        batch_transformation = torch.unsqueeze(batch_transformation, dim=1)
        # Part 3
        part3_batch_x = self.conv1(batch_transformation)
        part3_batch_x = self.conv2(part3_batch_x)
        part3_batch_x = part3_batch_x.permute(0, 3, 2, 1).flatten(2)
        part3_batch_x, _ = self.gru(part3_batch_x)
        batch_y = self.classify(part3_batch_x[:, -1])

        return batch_y

    def linear_activation(self, x):
        # k = 1
        # b = 0
        # y = k*x + b
        y = x
        return y


if __name__ == '__main__':
    model = PETCGDNN(11)
    model = model.cuda()
    input = torch.rand(1, 2, 128).cuda()
    # start = time.time()
    output = model(input)
    # end = time.time()
    # print((end-start)/400.0)
    # print(model)

    ### thop cal ###
    macs, params = profile(model, inputs=(input,))
    print(f"FLOPS: {macs / 1e6:.2f}M")
    print(f"params: {params / 1e3:.2f}K")
