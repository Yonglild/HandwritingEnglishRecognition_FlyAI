# -*- coding: utf-8 -*-
import argparse
import os

from flyai.data_helper import DataHelper
from flyai.framework import FlyAI
from flyai.utils.log_helper import train_log

from path import MODEL_PATH
from dataloader import Mydata
from models import net

import torch
from torch.utils.data import DataLoader
import time
'''
此项目为FlyAI2.0新版本框架，数据读取，评估方式与之前不同
2.0框架不再限制数据如何读取
样例代码仅供参考学习，可以自己修改实现逻辑。
模版项目下载支持 PyTorch、Tensorflow、Keras、MXNET、scikit-learn等机器学习框架
第一次使用请看项目中的：FlyAI2.0竞赛框架使用说明.html
使用FlyAI提供的预训练模型可查看：https://www.flyai.com/models
学习资料可查看文档中心：https://doc.flyai.com/
常见问题：https://doc.flyai.com/question.html
遇到问题不要着急，添加小姐姐微信，扫描项目里面的：FlyAI小助手二维码-小姐姐在线解答您的问题.png
'''
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

# 项目的超参，不使用可以删除
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=6, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=1, type=int, help="batch size")
args = parser.parse_args()

use_gpu = torch.cuda.is_available()
if use_gpu:
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
print(device)
num_step = 0

#TODO label_smoothing

class Main(FlyAI):
    '''
    项目中必须继承FlyAI类，否则线上运行会报错。
    '''

    def download_data(self):
        # 根据数据ID下载训练数据
        data_helper = DataHelper()
        data_helper.download_from_ids("HandwritingEnglishRecognition")

    def deal_with_data(self):
        train_data = Mydata("train", ratio=0.99, train_transform=True)
        val_data = Mydata("val", ratio=0.99, val_transform=True)
        train_loader = DataLoader(train_data, batch_size=args.BATCH, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=args.BATCH, shuffle=False)
        return train_loader, val_loader

    def train(self):
        global num_step
        flag = 0
        # model_name = ["resnet", "densenet", "efficientnet"]
        model_name = ["resnet"]
        model_list = net(num_class=28, model_name=model_name)
        for m in range(len(model_list)):
            max_correct = 0
            model = model_list[m].to(device)

            # loss
            loss_fun = torch.nn.CrossEntropyLoss()
            print(torch.__version__)
            print(torch.version.cuda)
            for epoch in range(args.EPOCHS):
                # scheduler.step(epoch)
                model.train()
                lr = 1e-3
                if epoch >= 2:
                    lr = 1e-4
                if epoch > 6:
                    lr = 2.5e-5
                if epoch > 8:
                    lr = 1e-5
                # optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=3e-4)
                optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=1e-4, nesterov=True)

                sum_loss = 0
                correct = 0
                sum = 0

                train_loader, val_loader = self.deal_with_data()
                start_time = 0
                for i, (roi_imgs, labels) in enumerate(train_loader):
                    if -1 in labels:
                        continue
                    if len(roi_imgs) != labels.shape[1]:
                        continue

                    time3 = time.time()
                    # print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                    # print('data_loader:', time3-start_time)
                    inputs = torch.cat(roi_imgs, dim=0).permute(0, 3, 1, 2)

                    sum += 1
                    num_step += 1
                    inputs, labels = inputs.to(device), labels.to(device).squeeze()
                    optimizer.zero_grad()

                    # time0 = time.time()
                    output = model(inputs)
                    if labels.ndim == 0 or output.ndim == 0:
                        continue
                    if output.shape[0] != labels.shape[0]:
                        continue
                    # print(output.size(), labels.size())

                    time1 = time.time()
                    # print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                    # print('model(inputs):', time1 - time3)

                    loss = loss_fun(output, labels)

                    loss.backward()
                    optimizer.step()

                    time2 = time.time()
                    # print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                    # print('loss_back & optim:', time2-time1)

                    sum_loss += loss.item()
                    predict = torch.max(output, 1)[1]
                    tmp = (predict == labels).sum().item()
                    if tmp == len(roi_imgs):
                        correct += 1

                    rate = (i + 1) / len(train_loader) * 100
                    print('rate:{} sum:{} lr:{}'.format(int(rate), sum, lr))
                    start_time = time.time()
                    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                    # print('get_result:', start_time - time2)
                    # print('all_time:', start_time-time3)
                    # if rate == 60 and flag == 0:
                    #     lr = 1e-4
                    #     for param_group in optimizer.param_groups:
                    #         param_group['lr'] = lr
                    #     flag += 1
                    # if rate == 85 and flag == 1:
                    #     lr = 1e-5
                    #     for param_group in optimizer.param_groups:
                    #         param_group['lr'] = lr
                    #     flag += 1

                train_loss = sum_loss / sum
                train_acc = correct / sum

                #############validation###########
                print('validation!!!!!!!!!!!!')
                model.eval()
                sum_loss = 0
                sum = 0
                val_acc = 0
                correct = 0

                for i, (roi_imgs, labels) in enumerate(val_loader):
                    if -1 in labels:
                        continue

                    if len(roi_imgs) != labels.shape[1]:
                        continue

                    inputs = torch.cat(roi_imgs, dim=0).permute(0, 3, 1, 2)

                    sum += 1
                    inputs, labels = inputs.to(device), labels.to(device).squeeze()
                    output = model(inputs)
                    if labels.ndim == 0 or output.ndim == 0:
                        continue
                    if output.shape[0] != labels.shape[0]:
                        continue
                    sum_loss += loss_fun(output, labels).item()
                    predict = torch.max(output, 1)[1]
                    tmp = (predict == labels).sum().item()
                    if tmp == len(roi_imgs):
                        correct += 1

                    val_acc = correct / sum

                print('val sum:', sum)
                print('val_acc:', val_acc)
                if val_acc >= max_correct:
                    max_correct = val_acc
                    # torch.save(model, os.path.join("COVIDC_densenet121_best.pth"))
                    torch.save(model, os.path.join(MODEL_PATH, model_name[m] + "_best.pth"))
                    # torch.save(model, os.path.join(MODEL_PATH, str(epoch)+".pth"))
                    # sava_train_model("COVIDC_densenet121_best.pth", overwrite=True, dir_name="lab_model")

                val_loss = sum_loss / sum
                val_acc = correct / sum
                train_log(train_loss=train_loss, train_acc=train_acc, val_loss=val_loss, val_acc=val_acc)
                print("epoch:", epoch, "train_loss:", train_loss, "train_acc:", train_acc, "val_loss:", val_loss, "val_acc:", val_acc)


if __name__ == '__main__':
    main = Main()
    main.download_data()
    main.train()
