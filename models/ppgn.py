import torch
import torch.nn as nn


def diag_offdiag_meanpool(input, n_tensor):


    diag = torch.diagonal(input, dim1=-2, dim2=-1)  # BxS

    sum_diag = diag.sum(-1)
    mean_diag = sum_diag / n_tensor
    # with torch.no_grad():

    mean_offdiag = (input.sum(dim=(-2,-1)) - sum_diag) / (n_tensor*(n_tensor-1))  # BxS


    return torch.cat((mean_diag, mean_offdiag), dim=1) 

class PPGN_block(nn.Module):
    """
    Imputs: N x input_depth x m x m
    Take the input through 2 parallel MLP routes, multiply the result, and add a skip-connection at the end.
    At the skip-connection, reduce the dimension back to output_depth
    """
    def __init__(self, cfg, in_features, out_features, depth=2):
        super().__init__()

        self.mlp1 = MlpBlock(2*in_features, out_features, depth)
        self.mlp2 = MlpBlock(in_features, out_features, depth)
        self.skip = SkipConnection(in_features+out_features, out_features)
        

    def forward(self, inputs, mask, n_list):
        mlp1 = self.mlp1(torch.cat([inputs, inputs.transpose(-1,-2)], dim=1)) * mask
        mlp2 = self.mlp2(inputs) * mask

        mult = torch.matmul(mlp1, mlp2)
        out = self.skip(in1=inputs, in2=mult) * mask
        return out

class MlpBlock(nn.Module):
    """
    Block of MLP layers with activation function after each (1x1 conv layers).
    """
    def __init__(self, in_features, out_features, depth_of_mlp, activation_fn=nn.functional.relu):
        super().__init__()
        self.activation = activation_fn
        self.convs = nn.ModuleList()
        for i in range(depth_of_mlp):
            self.convs.append(nn.Conv2d(in_features, out_features, kernel_size=1, padding=0, bias=True))
            _init_weights(self.convs[-1])
            in_features = out_features

    def forward(self, inputs):
        out = inputs
        for conv_layer in self.convs:
            out = self.activation(conv_layer(out))

        return out

class SkipConnection(nn.Module):
    """
    Connects the two given inputs with concatenation
    :param in1: earlier input tensor of shape N x d1 x m x m
    :param in2: later input tensor of shape N x d2 x m x m
    :param in_features: d1+d2
    :param out_features: output num of features
    :return: Tensor of shape N x output_depth x m x m
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size=1, padding=0, bias=True)
        _init_weights(self.conv)

    def forward(self, in1, in2, in3=None):
        # in1: N x d1 x m x m
        # in2: N x d2 x m x m
        if in3 is not None:
            out = torch.cat((in1, in2, in3), dim=1)
        else:
            out = torch.cat((in1, in2), dim=1)
        out = self.conv(out)
        return out

class FullyConnected(nn.Module):
    def __init__(self, in_features, out_features, activation_fn=nn.functional.relu):
        super().__init__()

        self.fc = nn.Linear(in_features, out_features)
        _init_weights(self.fc)

        self.activation = activation_fn

    def forward(self, input):
        out = self.fc(input)
        if self.activation is not None:
            out = self.activation(out)

        return out

def _init_weights(layer):
    """
    Init weights of the layer
    :param layer:
    :return:
    """
    nn.init.xavier_uniform_(layer.weight)
    # nn.init.xavier_normal_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class PPGN(nn.Module):
    def __init__(self, cfg):
        """
        Build the model computation graph, until scores/values are returned at the end
        """
        super().__init__()

        self.cfg = cfg
        depth = cfg.depth  # List of number of features in each regular block
        input_dim = cfg.input_dim  # Number of features of the input
        hidden_dim = cfg.hidden_dim
        # First part - sequential mlp blocks
        self.reg_blocks = nn.ModuleList()
        for _ in range(depth):
            mlp_block = PPGN_block(cfg, input_dim+1, hidden_dim)
            self.reg_blocks.append(mlp_block)
            input_dim = hidden_dim

        # Second part
        self.fc_layers = nn.ModuleList()

            # Sequential fc layers

        self.fc_layers.append(FullyConnected(2*hidden_dim, cfg.suffix_dim))
        self.fc_layers.append(FullyConnected(cfg.suffix_dim, cfg.out_dim, activation_fn=None))

    def forward(self, input, mask, n_list):
        x = input

        for i, block in enumerate(self.reg_blocks):
            x = torch.cat([x, torch.eye(x.shape[-1]).float().repeat(x.shape[0],1,1,1).to(x.device)],dim=1)
            x = block(x, mask, n_list)
            x = x * mask

        x = diag_offdiag_meanpool(x, n_list)  
        for fc in self.fc_layers:
            x = fc(x)

        return x.squeeze()