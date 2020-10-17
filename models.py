import torchvision.models as models
import torch
# from efficientnet_pytorch import EfficientNet
# from torchsummary import summary
import torch.nn as nn


def net(num_class, model_name, pretrain=True):
    model_list = []

    for name in model_name:
        if name == "resnet":
            model = models.resnet18(pretrained=pretrain)
            model.fc = torch.nn.Sequential(torch.nn.Linear(512, num_class))
            model_list.append(model)
        # elif name == "densenet":
        #     model = models.densenet121(pretrained=pretrain)
        #     model = Densenet(model, num_class)
        #     # model.fc = torch.nn.Sequential(torch.nn.Linear(2048, 2))
        #     model_list.append(model)
        # elif name == "efficientnet":
        #     model = EfficientNet.from_pretrained('efficientnet-b5', num_classes=num_class)
        #     model_list.append(model)

        # model = models.densenet121(pretrained=pretrain)
        # model = EfficientNet.from_pretrained('efficientnet-b5')
        # model.conv1 = torch.nn.Sequential(torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False))
        # model.fc = torch.nn.Sequential(torch.nn.Linear(2048, num_class))
    return model_list


class Densenet(nn.Module):
    def __init__(self, model, num_class):
        super(Densenet, self).__init__()
        self.ori_model = model
        # self.relu = nn.ReLU(inplace=True)
        self.linear = nn.Linear(1000, num_class)

    def forward(self, x):
        x = self.ori_model(x)
        # x = self.relu(x)
        x = self.linear(x)
        return x


if __name__ == '__main__':
    use_GPU = torch.cuda.is_available()
    # model_name = ["resnet", "densenet", "efficientnet"]
    model_name = ["resnet"]
    model_list = net(num_class=2, model_name=model_name)
    # for i in range(len(model_list)):
    model = model_list[0]
    if use_GPU:
        device = torch.device("cuda")
        model = model.to(device)
    # print(summary(model, input_size=(3, 448, 448)))