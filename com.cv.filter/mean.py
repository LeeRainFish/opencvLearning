import numpy as np


def means_filter(input_image, filter_size):
    '''
    均值滤波器
    :param input_image: 输入图像
    :param filter_size: 滤波器大小
    :return: 输出图像

    注：此实现滤波器大小必须为奇数且 >= 3
    '''
    input_image_cp = np.copy(input_image)  # 输入图像的副本

    filter_template = np.ones((filter_size, filter_size))  # 空间滤波器模板

    pad_num = int((filter_size - 1) / 2)  # 输入图像需要填充的尺寸

    input_image_cp = np.pad(input_image_cp, (pad_num, pad_num), mode="constant", constant_values=0)  # 填充输入图像

    m, n = input_image_cp.shape  # 获取填充后的输入图像的大小

    output_image = np.copy(input_image_cp)  # 输出图像

    # 空间滤波
    for i in range(pad_num, m - pad_num):
        for j in range(pad_num, n - pad_num):
            output_image[i, j] = np.sum(filter_template * input_image_cp[i - pad_num:i + pad_num + 1, j - pad_num:j + pad_num + 1]) / (filter_size ** 2)

    output_image = output_image[pad_num:m - pad_num, pad_num:n - pad_num]  # 裁剪

    return output_image


def median_filter(input_image, filter_size):
    '''
    中值滤波器
    :param input_image: 输入图像
    :param filter_size: 滤波器大小
    :return: 输出图像

    注：此实现滤波器大小必须为奇数且 >= 3
    '''
    input_image_cp = np.copy(input_image)  # 输入图像的副本

    pad_num = int((filter_size - 1) / 2)  # 输入图像需要填充的尺寸

    input_image_cp = np.pad(input_image_cp, (pad_num, pad_num), mode="constant", constant_values=0)  # 填充输入图像

    m, n = input_image_cp.shape  # 获取填充后的输入图像的大小

    output_image = np.copy(input_image_cp)  # 输出图像

    # 空间滤波
    for i in range(pad_num, m - pad_num):
        for j in range(pad_num, n - pad_num):
            output_image[i, j] = np.median(input_image_cp[i - pad_num:i + pad_num + 1, j - pad_num:j + pad_num + 1])

    output_image = output_image[pad_num:m - pad_num, pad_num:n - pad_num]  # 裁剪

    return output_image

