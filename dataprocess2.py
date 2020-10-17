import numpy as np
import cv2
import os

def display(img, s):
    cv2.imshow(s, img)
    cv2.waitKey(0)

def projection(img):
    h, w = img.shape[:2]
    proj = np.zeors(h)
    for i in range(w):
        for j in range(h):
            if img[j, i] > 200:
                proj[j] += 1
    # print('proj:', proj)

def get_split_line(img, projection_row):
    split_line_list = []
    flag = False
    start = 0
    end = 0
    for i in range(0, len(projection_row)):
        if flag == False and projection_row[i] >= 600:
            flag = True
            start = i
        elif flag and projection_row[i] < 600:
            flag = False
            end = i
            if end - start < 9:
                continue
            else:
                split_line_list.append((start, end))
        elif flag and i == len(projection_row) - 1:
            flag = False
            end = i
            if end - start < 9:  # need specify or rewrite
                continue
            else:
                split_line_list.append((start, end))
    return split_line_list

# (x, y, w, h) 左上角点
def get_contours(img):
    contour_list = []
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        contour_list.append((x, y, w, h))
    # display(img, 'rectange')
    return contour_list

def sort_merge(contour_row):
    # contour_row = sorted(contour_row, key=lambda x: x[0])  # sort by x
    # print(contour_row)
    i = 0
    for _ in contour_row:    # 这部分的合并规则用的是刘成林老师paper中的方法
        if i == len(contour_row) - 1 or contour_row[i][0] == -1:
            break
        # print(contour_row[i])
        rectR = contour_row[i + 1]
        rectL = contour_row[i]
        if rectL[2] < 3 and (rectR[0] - rectL[0]) > 3 * rectL[2]:
            i += 1
            continue
        ovlp = rectL[0] + rectL[2] - rectR[0]
        dist = abs((rectR[0] + rectR[2] / 2) - (rectL[0] - rectL[2] / 2))       ## ???
        w_L = rectL[0] + rectL[2]
        w_R = rectR[0] + rectR[2]
        span = (w_R if w_R > w_L else w_L) - rectL[0]
        nmovlp = (ovlp / rectL[2] + ovlp / rectR[2]) / 2 - dist / span / 8
        if nmovlp > 0:
            x = rectL[0]
            y = (rectL[1] if rectL[1] < rectR[1] else rectR[1])
            w_L = rectL[0] + rectL[2]
            w_R = rectR[0] + rectR[2]
            w = (w_R if w_R > w_L else w_L) - x
            h_L = rectL[1] + rectL[3]
            h_R = rectR[1] + rectR[3]
            h = (h_R if h_R > h_L else h_L) - y
            contour_row[i] = (x, y, w, h)
            contour_row.pop(i + 1)  # after pop , index at i
            # contour_row.append((-1, -1, -1, -1))  # add to fix bug(the better way is use iterator)
            i -= 1
        i += 1
    # print(contour_row)
    return contour_row


def combine_verticalLine(contour_row):
    i = 0
    pop_num = 0
    for _ in contour_row:
        rect = contour_row[i]
        if rect[0] == -1:
            break

        if rect[2] == 0:
            i += 1
            continue

        if rect[3] * 1.0 / rect[2] > 4:
            if i != 0 and i != len(contour_row) - 1:
                rect_left = contour_row[i - 1]
                rect_right = contour_row[i + 1]
                left_dis = rect[0] - rect_left[0] - rect_left[2]
                right_dis = rect_right[0] - rect[0] - rect[2]
                # if left_dis <= right_dis:
                if left_dis <= right_dis and rect_left[2] < rect_right[2]:
                    x = rect_left[0]
                    y = (rect_left[1] if rect_left[1] < rect[1] else rect[1])
                    w = rect[0] + rect[2] - rect_left[0]
                    h_1 = rect_left[1] + rect_left[3]
                    h_2 = rect[1] + rect[3]
                    h_ = (h_1 if h_1 > h_2 else h_2)
                    h = h_ - y
                    contour_row[i - 1] = (x, y, w, h)
                    contour_row.pop(i)
                    contour_row.append((-1, -1, -1, -1))
                    pop_num += 1
                    # don't need recursive merge, causing it's left and right merge
                else:
                    x = rect[0]
                    y = (rect[1] if rect[1] < rect_right[1] else rect_right[1])
                    w = rect_right[0] + rect_right[2] - rect[0]
                    h_1 = rect_right[1] + rect_right[3]
                    h_2 = rect[1] + rect[3]
                    h_ = (h_1 if h_1 > h_2 else h_2)
                    h = h_ - y
                    contour_row[i] = (x, y, w, h)
                    contour_row.pop(i + 1)
                    contour_row.append((-1, -1, -1, -1))
                    pop_num += 1
        i += 1
    for i in range(0, pop_num):
        contour_row.pop()
    return contour_row


def split_oversizeWidth(contour_row):
    i = 0
    for _ in contour_row:
        rect = contour_row[i]
        if rect[2] * 1.0 / rect[3] > 1.2:  # height/width>1.2 -> split
            x_new = int(rect[0] + rect[2] / 2 + 1)
            y_new = rect[1]
            w_new = rect[0] + rect[2] - x_new
            h_new = rect[3]
            contour_row[i] = (rect[0], rect[1], int(rect[2] / 2), rect[3])
            contour_row.insert(i + 1, (x_new, y_new, w_new, h_new))
        i += 1
    return contour_row

def draw_contour(img, roi, roicon):
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    row_up, row_down = roi[:2]
    for con in roicon:
        x, y, w, h = con[:]
        cv2.rectangle(img, (x, row_up + y), (x + w, row_up + y + h), (100, 255, 0), 1)
    # display(roi, 'draw_con')
    return img

# 删除小的框,合并相交的框
# def del_small_con(roicon):
#     i, j = 0, 0
#     copy_roicon = roicon.copy()
#     tmp_l = len(copy_roicon)
#     for _ in range(tmp_l):
#         if i > tmp_l - 1:
#             break
#         tmp_roicon = copy_roicon[i]
#         if tmp_roicon[2] <= 3 and tmp_roicon[3] <= 3:
#             del roicon[j]
#             i += 1
#             continue
#         if i + 1 > tmp_l - 1:
#             break
#         tmp1_roicon = copy_roicon[i + 1]
#         # 合并相近的小框
#         if tmp_roicon[0] + tmp_roicon[2] > tmp1_roicon[0] + 3 or (tmp_roicon[0] + tmp_roicon[2] > tmp1_roicon[0] - 3 and (tmp_roicon[2] <= 4 or tmp1_roicon[2] <= 4)):
#             x, y = tmp_roicon[0], min(tmp1_roicon[1], tmp_roicon[1])
#             w = max(tmp_roicon[0] + tmp_roicon[2], tmp1_roicon[0] + tmp1_roicon[2]) - tmp_roicon[0]
#             h = max(tmp_roicon[1] + tmp_roicon[3], tmp1_roicon[1] + tmp1_roicon[3]) - y
#             roicon[j] = (x, y, w, h)
#             del roicon[j + 1]
#             i += 1
#         i += 1
#         j += 1
#     return roicon

def del_small_con(roicon):
    i, j = 0, 0
    copy_roicon = roicon.copy()
    tmp_l = len(copy_roicon)
    for _ in range(tmp_l):
        if i > tmp_l - 1:
            break
        tmp_roicon = copy_roicon[i]
        if tmp_roicon[2] <= 3 and tmp_roicon[3] <= 3:
            del roicon[j]
            i += 1
            continue
        if i + 1 > tmp_l - 1:
            break
        tmp1_roicon = copy_roicon[i + 1]
        # 合并相近的小框
        if tmp_roicon[0] + tmp_roicon[2] > tmp1_roicon[0] + 3 or (tmp_roicon[0] + tmp_roicon[2] > tmp1_roicon[0] - 3 and (tmp_roicon[2] <= 4 or tmp1_roicon[2] <= 4)):
            x, y = tmp_roicon[0], min(tmp1_roicon[1], tmp_roicon[1])
            w = max(tmp_roicon[0] + tmp_roicon[2], tmp1_roicon[0] + tmp1_roicon[2]) - tmp_roicon[0]
            h = max(tmp_roicon[1] + tmp_roicon[3], tmp1_roicon[1] + tmp1_roicon[3]) - y
            roicon[j] = (x, y, w, h)
            del roicon[j + 1]
            # i += 1
            copy_roicon[i+1] = (x, y, w, h)
            j -= 1
        i += 1
        j += 1
    return roicon

def cal_average(distance):
    pass


if __name__ == '__main__':
    dir = './data/input/HandwritingEnglishRecognition/image'
    save_path = './data/input/HandwritingEnglishRecognition/roi'
    files = os.listdir(dir)
    # pic_path = './data/input/HandwritingEnglishRecognition/image/413.jpg'
    import time
    start_time = time.time()
    for file in files:
        # file = '635.jpg'
        print(file)
        srcimg = cv2.imread(os.path.join(dir, file), 0)

        # display(srcimg, 'srcimg')
        h, w = srcimg.shape[:2]
        _, img = cv2.threshold(srcimg, 175, 255, cv2.THRESH_BINARY_INV)

        # display(img, 'img')
        projection_row = cv2.reduce(img, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32S)  # projection
        # print(projection_row)
        # [(5, 22)]
        split_line_list = get_split_line(img, projection_row)
        print(split_line_list)

        roi = None
        roilist, con_list = [], []

        # 1. 如果只有一行，没有冒号；有冒号
        # 2. 有多行，手写字符与冒号同行
        # 3. 有多行，没有冒号:看字符行距
        flag = False        # 有无找到冒号或者有无找到字符
        print('len(split_line_list):', len(split_line_list) )

        ###     改进：统一get_contours成list, 下面来取    ###
        if len(split_line_list) == 0:       # 图像为空
            flag = True
            # roi = img
            roi = [0, h-1]

        elif len(split_line_list) == 1:
            split_line = split_line_list[0]
            row_up, row_down = split_line[0], split_line[1]
            img_row = img[row_up:row_down, :]
            contour_row = get_contours(img_row)
            contour_row.sort(key=lambda x: x[0])
            print('contour_row:', contour_row)
            for i in range(0, len(contour_row) - 1):
                con0, con1 = contour_row[i], contour_row[i + 1]
                if con0[0] < 20 or con0[0] > w * 0.8:  # 冒号的位置>20或者小于w*0.8
                    continue
                if abs(con1[0] - con0[0]) <= 1 and con0[2] <= 3 and con1[2] <= 3 and con0[3] <= 3 and con1[3] <= 3:
                    tmp = contour_row[i - 1]
                    if (con0[0] - tmp[0] - tmp[2]) <= 3:  # 冒号与前一个字符相隔一定距离　
                        continue
                    if i + 2 < len(contour_row):  # 手写字符在冒号右边
                        hand_start = contour_row[i + 2][0]
                        # roi = img[row_up:row_down, hand_start:]
                        roi = [row_up, row_down]
                        # roi = img[row_up:row_down, :]
                        roilist.append(roi)
                        con_list.append(contour_row[i+2:])
                        flag = True     # 通过冒号找到
                        break
                    else:
                        # roi = img[row_up:row_down, :]     # 全白
                        roi = [row_up, row_down]
                        flag = True
            if flag == False:       # 没有冒号，则该行都是手写字符
                # roilist.append(img_row)
                roilist.append([row_up, row_down])
                con_list.append(contour_row)
                flag = True         # 这里表示，找到字符

        elif len(split_line_list)>1:
            for split_line in split_line_list:
                row_up, row_down = split_line[0], split_line[1]
                img_row = img[row_up:row_down, :]
                contour_row = get_contours(img_row)
                # display(img_row, 'img_row')
                # con_list.append(contour_row)
                contour_row.sort(key=lambda x:x[0])
                print('contour_row:', contour_row)

                hand_start = 0                       # 手写字符开始的位置
                for i in range(0, len(contour_row)-1):
                    con0, con1 = contour_row[i], contour_row[i + 1]
                    if con0[0] < 20 or con0[0] > w * 0.8:  # 冒号的位置>10
                        continue
                    if abs(con1[0] - con0[0]) <= 1 and con1[2] <= 3 and con0[2] <= 3 and con0[3] <= 3 and con1[3] <= 3:
                        tmp = contour_row[i-1]
                        if (con0[0] - tmp[0] - tmp[2]) <= 3:        # 冒号与前一个字符相隔一定距离　
                            continue
                        flag = True                 # 找到冒号
                        if i + 2 < len(contour_row):    # 手写字符在冒号右边
                            hand_start = contour_row[i + 2][0]
                            # roi = img[row_up:row_down, hand_start:]
                            # roi = img[row_up:row_down, :]
                            roi = [row_up, row_down]
                            roilist.append(roi)
                            con_list.append(contour_row[i+2:])
                            break

        # 多行，没找到字符或者冒号， 通过判断字符行距
        if flag == False:
            print('没有冒号')
            big_var, big_distance, small_num = 0, 0, 100
            roi, roicon = None, None
            for split_line in split_line_list:
                row_up, row_down = split_line[0], split_line[1]
                img_row = img[row_up:row_down, :]
                contour_row = get_contours(img_row)
                # display(img_row, 'img_row')
                # con_list.append(contour_row)
                contour_row.sort(key=lambda x:x[0])
                print('contour_row:', contour_row)
                distance = np.array([contour_row[i+1][0] - contour_row[i][0] - contour_row[i][2] for i in range(len(contour_row)-1)])
                # list_w = [contour_row[i][2] for i in range(len(contour_row))]
                # var_w = np.var(list_w)
                # 计算整齐度(方差)
                # 去掉最大值，计算平均值

                # 小于2的个数
                a = np.sum(distance<=2)
                if a < small_num:
                    roi = [row_up, row_down]
                    # big_var = var_w
                    roicon = contour_row

                # if var_w > big_var:
                #     roi = [row_up, row_down]
                #     big_var = var_w
                #     roicon = contour_row

            roilist.append(roi)
            con_list.append(roicon)

        if len(roilist) > 0:
            roi = roilist[-1]       # 手写字符一般在最后一行
            roicon = con_list[-1]
            print('roi_con:', roicon)
            # display(roi, 'hand')

            #############闭运算重新框################
            # row_up, row_down = roi[:2]
            # roi_img = img[row_up:row_down, roicon[0][0]:]
            # display(roi_img, 'roi_img')
            # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            # img = cv2.morphologyEx(roi_img, cv2.MORPH_OPEN, kernel)
            # display(roi_img, 'th_img')


            ###############框出手写字符##################
            roicon = sort_merge(roicon)
            # roicon = split_oversizeWidth(roicon)
            # roicon = combine_verticalLine(roicon)


            # 删除小的框　合并相交的框
            roicon = del_small_con(roicon)
            print('roi_con_gai:', roicon)


            # display(img, 'img')

            img = draw_contour(srcimg, roi, roicon)

        # imgname = os.path.splitext(file)[0]
        cv2.imwrite(os.path.join(save_path, file), img)

    end_time = time.time()
    nums = len(files)
    print(nums)
    average_time = (end_time-start_time)/nums
    print('average_time:', average_time)
