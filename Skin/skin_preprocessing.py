import os
import shutil
import json

def make_roi(label):
    '''
    label에서 n 번째 roi 좌표 값 추출
    '''
    label = list(label['labelingInfo'][0]['polygon']['location'][0].items())
    x = []
    y = []
    for i in range(len(ax)):
        if i % 2 == 0:
            x.append(ax[i][1])
        else:
            y.append(ax[i][1])
    x = list(map(int, x))
    y = list(map(int, y))

    return [(i, j) for i, j in zip(x, y)]

def masking(img, label, lst, n,i):
    '''
    image file에 label에 있는 roi 영역 crop
    '''
    image = np.array(img)
    roi = np.array(make_roi(label, n))
    mask = np.zeros((img.shape[0], img.shape[1]))

    cv2.fillConvexPoly(mask, roi, 1)
    mask = mask.astype(np.bool)

    out = np.zeros_like(image)
    out[mask] = image[mask]

    plt.imshow(out[:,:,::-1])
    if label['labelingInfo'][n]['polygon']['label'][:2] == 'A3':  # 'A1', 'A3', 'A7'등 필요한 것으로 변경
        os.chdir(save_dir + label['labelingInfo'][n]['polygon']['label'][:2] + '/')
        return plt.savefig('label_'+lst[i])
    else:
        return print('/')

first_path = '/mnt/hackerton/dataset/Dataset/'
os.chdir(first_path)


train_path = './Skin/Train/'
val_path = './Skin/Validation/'
label_path = './Skin/label_data/Train/'

A1_list = sorted(os.listdir(train_path + 'A1/'))
A3_list = sorted(os.listdir(train_path + 'A3/'))
A7_list = sorted(os.listdir(train_path + 'A7/'))

A1_label = sorted(os.listdir(label_path + 'A1/'))
A3_label = sorted(os.listdir(label_path + 'A3/'))
A7_label = sorted(os.listdir(label_path + 'A7/'))

for i in range(len(A1_list)):
    img = cv2.imread(train_path + os.listdir(train_path)[1] + '/' + A3_list[i])  # (train_path)[0] 에서 0이 A1, 1이 A3, 2가 A7
    with open(label_path + os.listdir(label_path)[1] + '/' + A3_label[i]) as f:
        label = json.load(f)

    for j in range(len(label['labelingInfo'])):
        masking(img, label, j, i)
