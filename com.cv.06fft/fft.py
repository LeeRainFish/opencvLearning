import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt


def cv_show(name, src):
    cv.imshow(name, src)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.abspath('.'))
    img_toma_dir = root_dir + '/res/toma.jpg'  # 根据项目所在路径，找到用例所在的相对项目的路径
    img = cv.imread(img_toma_dir, cv.IMREAD_GRAYSCALE)

    # fft简单演示
    img_float32 = np.float32(img)

    dft = cv.dft(img_float32, flags=cv.DFT_COMPLEX_OUTPUT)
    fftshift = np.fft.fftshift(dft)

    magnitude_spectrum = np.log(cv.magnitude(fftshift[:, :, 0], fftshift[:, :, 1])) * 20
    #
    # plt.subplot(121),plt.imshow(img,cmap="gray")
    # plt.title("input image"),plt.xticks([]),plt.yticks([])
    #
    # plt.subplot(122),plt.imshow(magnitude_spectrum,cmap="gray")
    # plt.title("magnitude_spectrum "),plt.xticks([]),plt.yticks([])
    #
    # plt.show()

    # 低通滤波

    rows, cols = img.shape
    rrows, ccols = int(rows / 2), int(cols / 2)

    # 设置掩模在中心位置
    mask = np.zeros((rows, cols, 2), dtype=np.uint8)
    mask[rrows - 30:rrows + 30, ccols - 30:ccols + 30] = 1

    # IDFT

    fshift = fftshift * mask
    i_fftshift = np.fft.ifftshift(fshift)

    img_back = cv.idft(i_fftshift)
    img_back = cv.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    plt.subplot(121), plt.imshow(img, cmap="gray")
    plt.title("input image"), plt.xticks([]), plt.yticks([])

    plt.subplot(122), plt.imshow(img_back, cmap="gray")
    plt.title("result "), plt.xticks([]), plt.yticks([])

    plt.show()
