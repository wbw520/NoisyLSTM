import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


########################################
# unit for U-net
########################################
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        x = torch.cat([x2, x1], dim=1)
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


#######################################################
#  for PSPNet
#######################################################
class BackNet(object):
    def __init__(self, model, pretrained=True):
        if model == "resnet50":
            self.base_model = models.resnet50(replace_stride_with_dilation=[False, 2, 4], pretrained=pretrained)
        if model == "resnet101":
            self.base_model = models.resnet101(replace_stride_with_dilation=[False, 2, 4], pretrained=pretrained)

    def back(self):
        return self.base_model


class PyramidPoolingModule(nn.Module):
    def __init__(self, args, in_dim, reduction_dim, setting, lstm=False):
        super(PyramidPoolingModule, self).__init__()
        self.lstm = lstm
        self.features = []
        self.args = args
        kernel_size = self.args.lstm_kernel
        layer = self.args.lstm_layer
        for s in setting:
            if lstm:
                self.features.append(nn.Sequential(
                    nn.AdaptiveAvgPool2d(s),
                    nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                    # nn.BatchNorm2d(reduction_dim, momentum=.95),
                    nn.ReLU(inplace=True)
                ))
            else:
                self.features.append(nn.Sequential(
                    nn.AdaptiveAvgPool2d(s),
                    nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(reduction_dim, momentum=.95),
                    nn.ReLU(inplace=True)
                ))
        self.features = nn.ModuleList(self.features)
        if lstm:
            self.CON_ls = nn.Sequential(
                LsConv(in_dim, hidden_dim=[int(in_dim)], kernel_size=(kernel_size, kernel_size),
                       num_layers=layer, merge=args.merge),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        x_size = x.size()
        if self.lstm:
            final_lstm = lstm_function(x, self.args.sequence_len)
            FL = self.CON_ls(final_lstm)
            merge = FL
            out = [merge]
            for f in self.features:
                out.append(F.interpolate(f(FL), x_size[2:], mode="bilinear"))
        else:
            out = [x]
            for f in self.features:
                out.append(F.interpolate(f(x), x_size[2:], mode="bilinear"))

        out = torch.cat(out, dim=1)
        return out


def lstm_function(x, sequence_len):
    train_list = list(x.split(sequence_len, dim=0))
    list_for_lstm = []
    for i in range(len(train_list)):
        list_for_lstm.append(torch.unsqueeze(train_list[i], dim=0))
    if len(list_for_lstm) == 1:
        final_lstm = list_for_lstm[0]
    else:
        final_lstm = torch.cat(list_for_lstm, dim=0)
    return final_lstm
################################################
# LSTM module
################################################
class LsConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, merge=True):
        super(LsConv, self).__init__()
        self.merge = merge
        self.LC = ConvLSTM(input_dim=input_dim, hidden_dim=hidden_dim, kernel_size=kernel_size, num_layers=num_layers)

    def forward(self, x):
        x, last_states = self.LC(x)
        if self.merge:
            return last_states[0][0]
        else:
            return x[0][0]


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2

        self.Wxi = nn.Conv2d(self.input_dim, self.hidden_dim, self.kernel_size, 1, self.padding, bias=self.bias)
        self.Whi = nn.Conv2d(self.hidden_dim, self.hidden_dim, self.kernel_size, 1, self.padding, bias=False)
        self.Wxf = nn.Conv2d(self.input_dim, self.hidden_dim, self.kernel_size, 1, self.padding, bias=self.bias)
        self.Whf = nn.Conv2d(self.hidden_dim, self.hidden_dim, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc = nn.Conv2d(self.input_dim, self.hidden_dim, self.kernel_size, 1, self.padding, bias=self.bias)
        self.Whc = nn.Conv2d(self.hidden_dim, self.hidden_dim, self.kernel_size, 1, self.padding, bias=False)
        self.Wxo = nn.Conv2d(self.input_dim, self.hidden_dim, self.kernel_size, 1, self.padding, bias=self.bias)
        self.Who = nn.Conv2d(self.hidden_dim, self.hidden_dim, self.kernel_size, 1, self.padding, bias=False)

        self.Wci = None
        self.Wcf = None
        self.Wco = None

    def forward(self, x, cur_state):
        h, c = cur_state

        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)
        return ch, cc

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        if self.Wci is None:
            self.Wci = torch.zeros(batch_size, self.hidden_dim, height, width, device=self.Wxi.weight.device)
            self.Wcf = torch.zeros(batch_size, self.hidden_dim, height, width, device=self.Wxi.weight.device)
            self.Wco = torch.zeros(batch_size, self.hidden_dim, height, width, device=self.Wxi.weight.device)
        else:
            assert height == self.Wci.size()[2], 'Input Height Mismatched!'
            assert width == self.Wci.size()[3], 'Input Width Mismatched!'
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.Wxi.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.Wxi.weight.device))


class ConvLSTM(nn.Module):

    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.
    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, [16], [(3, 3)], 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=True, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i]))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor:
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state:
           [h, c] has the structure of batch , hidden_dim, height, width
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is None:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](x=cur_layer_input[:, t, :, :, :], cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


# input = torch.rand((3, 6, 256, 56, 56))
# print(input.size())
# wbw = LsConv(256, hidden_dim=[128], kernel_size=(1, 1), num_layers=1)
# out = wbw(input)
# print(out.size())
