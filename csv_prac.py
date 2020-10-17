import pandas


ratio = 0.9
typ = 'train'
imgs = []
data_path = "data/input/HandwritingEnglishRecognition/"
read = pandas.read_csv(data_path + "train.csv")
length = int(len(read) * ratio)
for i in range(len(read[:length] if typ == "train" else read[length:])):
    imgs.append([read["image_path"][i], read["label"][i]])

label = imgs[6][1]
labels = [x for x in label]
# labels = ['s','o', ' ', 'i', 'a', 'm', ' ', 'i', ' ']
blank_nums = sum([x == ' ' for x in labels])
print(blank_nums)
print(labels)
print('done!')


A = 'A'
Z = 'Z'
print(ord(A)-ord(Z))