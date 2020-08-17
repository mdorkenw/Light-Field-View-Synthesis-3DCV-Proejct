import torch.nn as nn


class ConditionalNorm2d(nn.Module):
    def __init__(self, num_features, dic, num_groups=8):
        super().__init__()
        name = dic['norm']
        num_classes = dic['num_classes']
        self.num_features = num_features
        if name == 'BN' or name == 'batch':
            self.bn = nn.BatchNorm2d(num_features, affine=False, track_running_stats=dic['running_stats'])
        elif name == 'group' or name == 'Group':
            self.bn = nn.GroupNorm(num_groups, num_features, affine=False)
        elif name == 'instance':
            self.bn = nn.InstanceNorm2d(num_features, affine=False, track_running_stats=dic['running_stats'])
        else:
            raise NotImplementedError('Normalization Method not implemented: ', name)

        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
        self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0

    def forward(self, x, y):
        out = self.bn(x)
        gamma, beta = self.embed(y).chunk(2, 1)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        return out


class Norm2D(nn.Module):
    def __init__(self, num_features, dic, num_groups=16):
        super().__init__()
        name = dic['norm']
        self.num_features = num_features
        if name == 'BN' or name == 'batch':
            self.bn = nn.BatchNorm2d(num_features, affine=True, track_running_stats=dic['running_stats'])
        elif name == 'group' or name == 'Group':
            self.bn = nn.GroupNorm(num_groups, num_features, affine=True)
        elif name == 'instance':
            self.bn = nn.InstanceNorm2d(num_features, affine=True, track_running_stats=dic['running_stats'])
        else:
            raise NotImplementedError('Normalization Method not implemented: ', name)

    def forward(self, x):
        out = self.bn(x)
        return out


class mySequential(nn.Sequential):
    def forward(self, *inputs):
        if type(inputs) == tuple:
            c = inputs[1]
            inputs = inputs[0]
        for module in self._modules.values():
            if isinstance(module, ConditionalNorm2d):
                inputs = module(inputs, c)
            else:
                inputs = module(inputs)
        return inputs
