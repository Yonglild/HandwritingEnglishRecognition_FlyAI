import os
from PIL import Image
from torchvision.transforms import transforms
import torch
from torch.utils.data import Dataset, DataLoader
import pandas
from preprocess import process
import cv2

def transform(typ):
    if typ == "train":
        return transforms.Compose([
            transforms.Resize(112),
            # transforms.CenterCrop((218, 178)),
            # transforms.RandomRotation(30, resample=Image.BICUBIC,
            #                           expand=False, center=(100, 300)),
            transforms.ToTensor(),
            # transforms.Normalize([0.5], [0.5])
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    if typ == "val":
        return transforms.Compose([
            transforms.Resize(112),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

# gap 字符取字　间隙
def get_roi_img(img, roi, roicon, gap):
    imgh, imgw = img.shape[:2]
    row_up, row_down = roi[:2]
    roi_imgs = []
    for con in roicon:
        x, y, w, h = con[:]
        x1, x2, y1, y2 = max(x-gap, 0), min(x + w + gap, imgw-1), max(row_up - gap, 0), min(row_down + gap, imgh-1)
        # cv2.imshow(' ', img[y1:y2, x1:x2])
        # cv2.waitKey()
        roi_imgs.append(img[y1:y2, x1:x2])
    return roi_imgs

def roiimg_split(roi_imgs):
    # 最大的图片
    big_w, big_w_indx, mean_w = 0, 0, 0
    big_img = None
    if len(roi_imgs) < 3:
        return roi_imgs
    for i in range(len(roi_imgs)):
        img = roi_imgs[i]
        h, w = img.shape[:2]
        # print(w)
        # cv2.imshow(' ', img)
        # cv2.waitKey()
        mean_w += w
        if w > big_w:
            big_w = w
            big_img = img
            big_w_indx = i
    mean_w = (mean_w - big_w)/(len(roi_imgs)-1)
    tmp = big_w / mean_w
    if tmp >= 1.8 and tmp < 2.5:
        _, w = big_img.shape[:2]
        img0 = big_img[:, :int(w/2)]
        img1 = big_img[:, int(w/2):]
        del roi_imgs[big_w_indx]
        roi_imgs.insert(big_w_indx, img0)
        roi_imgs.insert(big_w_indx+1, img1)
    elif tmp >= 2.5:
        _, w = big_img.shape[:2]
        step = int(w/3)
        img0 = big_img[:, :step]
        img1 = big_img[:, step:2*step]
        img2 = big_img[:, 2*step:]
        del roi_imgs[big_w_indx]
        roi_imgs.insert(big_w_indx, img0)
        roi_imgs.insert(big_w_indx+1, img1)
        roi_imgs.insert(big_w_indx+2, img2)
    return roi_imgs

# 找出空格及位置
def find_space(roicon):
    if len(roicon) < 5:
        return None
    big_space, big_indx, mean_space = 0, 0, 0
    for i in range(len(roicon)-1):
        con0, con1 = roicon[i], roicon[i+1]
        x0, _, w0, _ = con0[:]
        x1, _, w1, _ = con1[:]
        space = x1 - x0 - w0
        mean_space += space
        if space > big_space:
            big_space = space
            big_indx = i
    mean_space = (mean_space - big_space)/(len(roicon)-1)
    if mean_space < 5:
        return None
    tmp = big_space / mean_space
    if tmp >= 2.5 and len(roicon) > 5:
        return big_indx+1
    else:
        return None


class Mydata(Dataset):
    def __init__(self, typ, ratio, train_transform=True, val_transform=True):
        self.typ = typ
        self.ratio = ratio
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.data_path = "data/input/HandwritingEnglishRecognition/"
        self.imgs = []

        read = pandas.read_csv(self.data_path+"train.csv")
        length = int(len(read)*ratio)
        for i in range(len(read[:length] if self.typ=="train" else read[length:])):
            self.imgs.append([read["image_path"][i], read["label"][i]])

    def __getitem__(self, item):
        img_path = self.imgs[item][0]
        # print(img_path)
        label = self.imgs[item][1]
        srcimg = cv2.imread(os.path.join(self.data_path, img_path), 1)
        img = cv2.cvtColor(srcimg, cv2.COLOR_BGR2GRAY)
        # 分割手写字符
        roi, roicon = process(img)      # [row_up, row_down], [(), (), ...()]
        if roi is None:
            return -1, -1

        labels = [x for x in label]     # ['F', 'E', 'R', 'N', 'A', 'N', 'D', 'E', 'Z', ' ', 'B', 'L', 'A', 'N', 'C', 'O']

        # 空格如何得到(或者不计空格)
        roi_imgs = get_roi_img(srcimg, roi, roicon, gap=2)

        for i in range(len(labels)):
            if labels[i] == '-':
                labels[i] = 26
                continue
            elif labels[i] == ' ':
                labels[i] = 26
                continue
            elif labels[i] == "'":
                labels[i] = 27
                continue
            tmp = ord(labels[i]) - ord('A')
            if tmp >=32 and tmp<=57:    # 小写
                labels[i] = tmp - 32
            else:
                labels[i] = tmp

        if self.train_transform or self.val_transform:
            for i in range(len(roi_imgs)):
                roi_imgs[i] = torch.FloatTensor(cv2.resize(roi_imgs[i], (40, 56), cv2.INTER_LANCZOS4))

        return roi_imgs, torch.tensor(labels)

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    data_path = "data/input/HandwritingEnglishRecognition/"
    data = Mydata("train", 0.9, train_transform=True)
    dataloader = DataLoader(data, batch_size=1, shuffle=True)
    train_transform = True
    for i, (roi_imgs, labels) in enumerate(dataloader):
        if -1 in labels:
            continue
        print('label', labels)
        print(len(roi_imgs), len(labels))
        if len(roi_imgs) != len(labels):
            continue


        inputs = torch.cat(roi_imgs, dim=0).permute(0, 3, 1, 2)


        print('labels:', labels)
