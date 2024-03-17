import cv2


# 计算交并比
def compute_IOU(A, B):
    intersection = (A * B).sum()
    union = A.sum() + B.sum() - intersection
    IOU = intersection / union

    return IOU


def main():
    GT = cv2.imread("./3_5.png") / 255
    P1 = cv2.imread("./3.png") / 255
    print(compute_IOU(P1, GT))


if __name__ == '__main__':
    main()
