import os


def data2txt(data_rootdir):
    # 读取两个文件夹的所有图像并判断是否相等
    images = os.listdir(data_rootdir + 'data/')
    labels = os.listdir(data_rootdir + 'gt/')
    masks = os.listdir(data_rootdir+'mask/')
    images.sort()
    labels.sort()
    masks.sort()

    image_len = len(images)
    label_len = len(labels)
    mask_len = len(masks)

    assert image_len == label_len

    # 打开文本并写入路径
    trainText = open(data_rootdir + 'train800.txt', 'w')
    for i in range(image_len):
        image_dir = data_rootdir + 'data/' + images[i] + ' '
        label_dir = data_rootdir + 'gt/' + labels[i] + ' '
        mask_dir = data_rootdir + 'mask/' + masks[i] + '\n'

        trainText.write(image_dir)
        trainText.write(label_dir)
        trainText.write(mask_dir)

    trainText.close()
    print('finished!')


if __name__ == '__main__':
    data2txt('./data/training_data800/')