import torch.nn as nn, torch
from efficientnet.model import EfficientNet


class Net(nn.Module):
    def __init__(self, dic):
        super().__init__()
        self.attention = dic['attention_mask']
        if dic['input_domain'] == 'RGB':
            self.model = EfficientNet.from_pretrained('efficientnet-b0')
        else:
            self.model = EfficientNet.from_name('efficientnet-b0')
        # 1280 is the number of neurons in last layer. is diff for diff. architecture
        # self.dense_output = nn.Linear(2304, 512)
        self.dense_output = nn.Linear(1280, 512)
        # self.dense_output = nn.Linear(1280, dic["n_classes"])
        self.output = nn.Linear(512, dic["n_classes"])
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, mask):
        if not self.attention:
            mask = None
        feat = self.model.extract_features(x, mask)
        feat = nn.functional.avg_pool2d(feat, feat.size()[2:]).reshape(-1, 1280)
        return self.output(self.act(self.dense_output(feat)))


if __name__ == '__main__':
    import os, argparse
    import auxiliaries as aux

    os.environ["CUDA_VISIBLE_DEVICES"] = '9'

    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--config", type=str,
                        default='/export/home/mdorkenw/code/ALASKA2/network_base_setup.txt',
                        help="Define config file")
    args = parser.parse_args()
    opt = aux.extract_setup_info(args.config)[0]
    model = Net(opt.Network).cuda()
    print("Number of parameters in generator", sum(p.numel() for p in model.parameters()))
    dummy = torch.ones((2, 3, 512, 512)).cuda()
    breakpoint()
    print(model(dummy).shape)
