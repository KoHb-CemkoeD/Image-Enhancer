import math
from os import path, listdir
from time import time, process_time
from timeit import default_timer as timer

import cv2
import numpy as np

from processing.models.realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

INPUTS_FOLDER = 'inputs'
kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])

edsr = cv2.dnn_superres.DnnSuperResImpl_create()
edsr.readModel('processing/models/EDSR/EDSR_x4.pb')
edsr.setModel('edsr', 4)

fsrcnn = cv2.dnn_superres.DnnSuperResImpl_create()
fsrcnn.readModel('processing/models/FSRCNN/FSRCNN_x4.pb')
fsrcnn.setModel('fsrcnn', 4)

model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
esrgan = RealESRGANer(
    scale=4,
    model_path='processing/models/ESRGAN/RealESRGAN_x4plus.pth',
    model=model,
    tile=512,
    tile_pad=10,
    pre_pad=0,
    half=False)


def interpolate(img, res_img, h, w):
    inter_st = timer()
    inter_result = cv2.resize(res_img, dsize=(w, h), interpolation=cv2.INTER_NEAREST)
    print(' INTER_NEAREST time', timer() - inter_st, 'psnr', psnr(img, inter_result))

    inter_st = timer()
    inter_result = cv2.resize(res_img, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
    print(' INTER_LINEAR time', timer() - inter_st, 'psnr', psnr(img, inter_result))

    inter_st = timer()
    inter_result = cv2.resize(res_img, dsize=(w, h), interpolation=cv2.INTER_CUBIC)
    print(' INTER_CUBIC time', timer() - inter_st, 'psnr', psnr(img, inter_result))

    inter_st = timer()
    inter_result = cv2.resize(res_img, dsize=(w, h), interpolation=cv2.INTER_LANCZOS4)
    print(' INTER_LANCZOS4 time', timer() - inter_st, 'psnr', psnr(img, inter_result))


def kernel_filter(img, res_img):
    kernel_st = timer()
    kernel_result = cv2.GaussianBlur(res_img, ksize=(5, 5), sigmaX=1.0, sigmaY=1.0)
    print(' GaussianBlur time', timer() - kernel_st, 'psnr', psnr(img, kernel_result))

    kernel_st = timer()
    kernel_result = cv2.filter2D(res_img, -1, kernel)
    print(' filter2D time', timer() - kernel_st, 'psnr', psnr(img, kernel_result))

    kernel_st = timer()
    kernel_result = cv2.bilateralFilter(res_img, 9, 75, 75)
    print(' bilateralFilter time', timer() - kernel_st, 'psnr', psnr(img, kernel_result))


def ml_model(img, res_img):
    ml_st = timer()
    ml_result = edsr.upsample(res_img)
    print(' edsr time', timer() - ml_st, 'psnr', psnr(img, ml_result))

    ml_st = timer()
    ml_result = fsrcnn.upsample(res_img)
    print(' fsrcnn time', timer() - ml_st, 'psnr', psnr(img, ml_result))

    ml_st = timer()
    ml_result = esrgan.enhance(res_img)[0]
    print(' ml time', timer() - ml_st, 'psnr', psnr(img, ml_result))


def test():
    if path.isdir(INPUTS_FOLDER):
        print(INPUTS_FOLDER)
        img_paths = [path.join(INPUTS_FOLDER, f) for f in listdir(INPUTS_FOLDER) if
                     path.isfile(path.join(INPUTS_FOLDER, f))]
        print('image:', img_paths)
        for img_path in img_paths:
            print(img_path)
            t_img: np.ndarray = cv2.imread(img_path, 3)
            if t_img is None:
                continue
            img = t_img.astype(np.float32)

            h, w, _ = img.shape
            res_img = cv2.resize(img.astype(np.float32), (w // 4, h // 4), interpolation=cv2.INTER_AREA)
            interpolate(img, res_img, h, w)
            kernel_filter(img, cv2.resize(res_img, (w, h), interpolation=cv2.INTER_NEAREST))
            ml_model(cv2.resize(img, (res_img.shape[1] * 4, res_img.shape[0] * 4)), res_img)
            print()
    else:
        print('Folder', INPUTS_FOLDER, 'no exist!')


def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


if __name__ == '__main__':
    test()
