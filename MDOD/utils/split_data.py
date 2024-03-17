import os
import random
import cv2


def main():
    random.seed(0)  # 设置随机种子，保证随机结果可复现

    files_path = "../pre-dataset/output"
    assert os.path.exists(files_path), "path: '{}' does not exist.".format(files_path)

    val_rate = 0.2

    files_name = sorted([file.split(".")[0] for file in os.listdir(files_path)])
    files_num = len(files_name)
    val_index = random.sample(range(0, files_num), k=int(files_num * val_rate))
    train_files = []
    val_files = []
    for index, file_name in enumerate(files_name):
        if index in val_index:
            val_files.append(file_name)
        else:
            train_files.append(file_name)
    TR_path = "../dataset/DUTS-TR"
    TE_path = "../dataset/DUTS-TE"
    # os.mkdir(TR_path + '/DUTS-TR-Image')
    # os.mkdir(TR_path + '/DUTS-TR-Mask')
    # os.mkdir(TE_path + '/DUTS-TE-Image')
    # os.mkdir(TE_path + '/DUTS-TE-Mask')
    for i in train_files:
        img = cv2.imread('../pre-dataset/output/{}.png'.format(i))
        cv2.imwrite(TR_path + '/DUTS-TR-Mask/{}.png'.format(i), img)
        img2 = cv2.imread('../pre-dataset/images/{}.png'.format(i))
        cv2.imwrite(TR_path + '/DUTS-TR-Image/{}.png'.format(i), img2)
    for i in val_files:
        img = cv2.imread('../pre-dataset/output/{}.png'.format(i))
        cv2.imwrite(TE_path + '/DUTS-TE-Mask/{}.png'.format(i), img)
        img2 = cv2.imread('../pre-dataset/images/{}.png'.format(i))
        cv2.imwrite(TE_path + '/DUTS-TE-Image/{}.png'.format(i), img2)
    print("处理完毕")


if __name__ == '__main__':
    main()
