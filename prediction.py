# -*- coding: utf-8 -*
from flyai.framework import FlyAI
from PIL import Image
import numpy as np
import torch
import os
import cv2
import glob
from models import net
from path import *
from dataloader import get_roi_img, roiimg_split, find_space
from preprocess import process

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Prediction(FlyAI):
    def load_model(self):
        '''
        模型初始化，必须在此方法中加载模型
        '''
        self.model_name = ["resnet"]  #, "densenet", "efficientnet"]
        self.model_list = []
        for name in self.model_name:
            model = torch.load(os.path.join(MODEL_PATH, name+"_best.pth"))
            self.model_list.append(model)
            # model = torch.load("lab_model/pretrain_model/COVIDC_densenet121_best.pth")
            # model = ttach.SegmentationTTAWrapper(model, ttach.aliases.d4_transform(), merge_mode='mean')
            # self.model_list = model.to(device)
    def predict(self, image_path):
        '''
        模型预测返回结果
        :param input:  评估传入样例 {"image_path":"image\/172691.jpg"}
        :return: 模型预测成功之后返回给系统样例 {"label":"ZASSEOR"}
        '''

        pred = ''
        for n in range(len(self.model_list)):
            model = self.model_list[n]
            model = model.to(device)
            # img = Image.open(image_path).convert("RGB")
            srcimg = cv2.imread(image_path, 1)
            img = cv2.cvtColor(srcimg, cv2.COLOR_BGR2GRAY)
            # 分割手写字符
            roi, roicon = process(img)  # [row_up, row_down], [(), (), ...()]
            if roi is None:
                return {"label": "None"}

            roi_imgs = get_roi_img(srcimg, roi, roicon, gap=2)
            if len(roi_imgs) == 0:
                return {"label": "None"}
            # roi_imgs 拆分
            roi_imgs = roiimg_split(roi_imgs)

            big_indx = None
            big_indx = find_space(roicon)

            # roi_imgs 检查空格
            for i in range(len(roi_imgs)):
                # cv2.imshow('' ,cv2.resize(roi_imgs[i], (40, 56), cv2.INTER_LANCZOS4))
                # cv2.waitKey()
                roi_imgs[i] = torch.FloatTensor(cv2.resize(roi_imgs[i], (40, 56), cv2.INTER_LANCZOS4))
            inputs = torch.stack(roi_imgs, dim=0).permute(0, 3, 1, 2).to(device)

            inputs = inputs.to(device)
            output = model(inputs)
            if output.ndim == 0:
                return {"label": "None"}
            predict = torch.max(output, 1)[1]
            for i in range(predict.shape[0]):
                if predict[i] < 26:
                    pred += chr(predict[i].item() + 65)
                elif predict[i] == 26:
                    pred += '-'
                elif predict[i] == 27:
                    pred += "'"
            if big_indx is not None:
                l_pred = list(pred)
                l_pred.insert(big_indx, ' ')
                pred = ''.join(l_pred)
        return {"label": pred}
        # return {"label":"ZASSEOR"}

if __name__ == '__main__':
    # img_path = './data/input/HandwritingEnglishRecognition/image/161365.jpg'
    # path_list = glob.glob('./data/input/HandwritingEnglishRecognition/image/*.jpg')
    Prediction = Prediction()
    sum = 0
    correct, val_acc = 0, 0
    import pandas
    data_path = "data/input/HandwritingEnglishRecognition/"
    imgs = []

    Prediction.load_model()

    read = pandas.read_csv(data_path + "train.csv")
    length = int(len(read) * 0.95)
    for i in range(len(read[:length])):
        imgs.append([read["image_path"][i], read["label"][i]])
    for img_path, label in imgs:
        print(img_path)
        predict = Prediction.predict(os.path.join(data_path, img_path))
        print(predict)

        if predict['label'] is not None:
            sum += 1
            print('sum:{}/{}'.format(sum, length))
            if predict['label'] == label:
                correct += 1

        val_acc = correct / sum
    print('val_acc:', val_acc)