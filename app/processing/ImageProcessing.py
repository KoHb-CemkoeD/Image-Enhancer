from threading import Thread
from time import time, sleep

import gc
import cv2
import numpy as np
import torch

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
from basicsr.archs.rrdbnet_arch import RRDBNet

from processing.models.realesrgan import RealESRGANer

SCALE_FACTOR, INTER_METHOD, KERNEL_METHOD, ML_MODEL = 0, 1, 2, 3
INTERPOLATED_READY, KERNEL_READY, ML_READY = 0, 1, 2

INTER_NEAREST, INTER_LINEAR, INTER_CUBIC, INTER_LANCZOS = cv2.INTER_NEAREST, cv2.INTER_LINEAR, \
                                                          cv2.INTER_CUBIC, cv2.INTER_LANCZOS4
GAUSSIAN_BLUR, SHARPEN_FILTER, BILATERAL_FILTER = 0, 1, 2
FSRCNN, EDSR, ESRGAN = 0, 1, 2

# torch.cuda.is_available = lambda: False


class ProcessingThread(QThread):
    image_saved = pyqtSignal(QPixmap)
    param_changed = pyqtSignal(int)
    change_preview = pyqtSignal(int, QPixmap)

    def __init__(self):
        QThread.__init__(self)
        self.process_device = self.edsr = self.fsrcnn = self.esrgan = self.image = self.image_b = self.full_saving = None
        self.load_models()
        self.ml_models = [self.fsrcnn.upsample, self.edsr.upsample, self.esrgan_model]
        self.scale_fr, self.inter_md, self.kernel_md, self.ml_model = 4, cv2.INTER_NEAREST, GAUSSIAN_BLUR, self.fsrcnn.upsample

    def load_models(self):
        self.edsr = cv2.dnn_superres.DnnSuperResImpl_create()
        self.edsr.readModel('processing/models/EDSR/EDSR_x4.pb')
        self.edsr.setModel('edsr', 4)

        self.fsrcnn = cv2.dnn_superres.DnnSuperResImpl_create()
        self.fsrcnn.readModel('processing/models/FSRCNN/FSRCNN_x4.pb')
        self.fsrcnn.setModel('fsrcnn', 4)

        self.load_esrgan()

    def load_esrgan(self, tile=0):
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        self.esrgan = RealESRGANer(
            scale=4,
            model_path='processing/models/ESRGAN/RealESRGAN_x4plus.pth',
            model=model,
            tile=tile,
            tile_pad=10,
            pre_pad=0,
            half=False)

    def esrgan_model(self, img):
        output, _ = self.esrgan.enhance(img, outscale=4)
        return output

    def qpixmap_to_array(self, pixmap):
        q_img = pixmap.toImage()

        shape = (q_img.height(), q_img.bytesPerLine() * 8 // q_img.depth(), 4)
        ptr = q_img.bits()
        ptr.setsize(q_img.byteCount())
        result = np.array(ptr, dtype=np.uint8).reshape(shape)
        return result[..., :3]

    def nparray_to_qpixmap(self, img):
        w, h, ch = img.shape
        if img.ndim == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        q_img = QImage(img.data.tobytes(), h, w, 3 * h, QImage.Format_BGR888)
        return QPixmap(q_img)

    def init_full_saving(self, method, img):
        self.image_b = self.image
        self.set_pixmap(img)
        if method == INTER_METHOD:
            self.full_saving = self.run_inter
        elif method == KERNEL_METHOD:
            self.full_saving = self.run_kernel
        elif method == ML_MODEL:
            self.full_saving = self.run_ml
        print(self.full_saving)

    def run(self):
        w, h, _ = self.image.shape
        if self.full_saving is None:
            self.change_preview.emit(INTERPOLATED_READY, self.nparray_to_qpixmap(self.run_inter()))
            self.change_preview.emit(KERNEL_READY, self.nparray_to_qpixmap(self.run_kernel()))
            self.change_preview.emit(ML_READY, self.nparray_to_qpixmap(self.run_ml()))
        else:
            if w > 640 or h > 640:
                self.clear_cache()
            try:
                self.image_saved.emit(self.nparray_to_qpixmap(self.full_saving()))
            except Exception as e:
                for tiles in [448, 384, 320, 256, 192, 128, 64]:
                    self.clear_cache(tiles)
                    self.load_esrgan(tile=tiles)
                    try:
                        print(f'Trying clip image in {tiles} px tiles...')
                        self.image_saved.emit(self.nparray_to_qpixmap(self.full_saving()))
                    except Exception as e:
                        print('Err', e)
                        continue
                    break
            self.full_saving = None
            self.image = self.image_b
            self.clear_cache()
            self.load_esrgan()

    def clear_cache(self, tiles=512):
        self.esrgan = None
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        sleep(0.1)
        self.load_esrgan(tile=tiles)

    def run_inter(self):
        inter_st = time()
        inter_result = cv2.resize(self.image, None, fx=self.scale_fr, fy=self.scale_fr,
                                  interpolation=self.inter_md)
        print('inter time', time() - inter_st)
        return inter_result

    def run_kernel(self):
        kernel_st = time()
        kernel_result = None
        if self.kernel_md == GAUSSIAN_BLUR:
            kernel_result = cv2.GaussianBlur(self.image, ksize=(5, 5), sigmaX=1.0, sigmaY=1.0)
        elif self.kernel_md == SHARPEN_FILTER:
            kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
            kernel_result = cv2.filter2D(self.image, -1, kernel)
        elif self.kernel_md == BILATERAL_FILTER:
            kernel_result = cv2.bilateralFilter(self.image, 9, 75, 75)
        print('kernel time', time() - kernel_st)
        return kernel_result

        # result = self.edsr.upsample(img)
        # result = self.fsrcnn.upsample(img)
        # result = self.esrgan_pr(img)
        # result = cv2.bilateralFilter(, 9, 75, 75);
        # result = cv2.GaussianBlur(img, ksize=(5, 5), sigma=1.0)
        # result = self.unsharp_mask(img)

    def run_ml(self):
        ml_st = time()
        ml_result = self.ml_model(self.image)
        print('ml time', time() - ml_st)
        return ml_result



    def set_pixmap(self, pixmap):
        self.image = self.qpixmap_to_array(pixmap)
        # self.image: np.ndarray
        # print(self.image.shape)
        # h = self.image.shape[0]
        # half = h // 2
        # f_half: np.ndarray = self.image[:half, ]
        # s_half = self.image[half:, ]
        # fs = np.vstack([f_half, s_half])
        # print('fh', f_half)
        # print('sh', s_half)
        # print('fs', fs.shape)

    def unsharp_mask(self, image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
        # kernel = np.array([[-1, -1, -1],
        #                    [-1, 9, -1],
        #                    [-1, -1, -1]])
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])

        sharpened = cv2.filter2D(image, -1, kernel)
        # blurred = cv2.GaussianBlur(image, kernel_size, sigma)
        # sharpened = float(amount + 1) * image - float(amount) * blurred
        # sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
        # sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
        # sharpened = sharpened.round().astype(np.uint8)
        # if threshold > 0:
        #     low_contrast_mask = np.absolute(image - blurred) < threshold
        #     np.copyto(sharpened, image, where=low_contrast_mask)
        return sharpened

    def change_param(self, param_type, value):
        if param_type == SCALE_FACTOR or param_type == INTER_METHOD:
            self.inter_md = value
            Thread(target=lambda: self.change_preview.emit(INTERPOLATED_READY,
                                                           self.nparray_to_qpixmap(self.run_inter()))).start()
        elif param_type == KERNEL_METHOD:
            self.kernel_md = value
            Thread(target=lambda: self.change_preview.emit(KERNEL_READY,
                                                           self.nparray_to_qpixmap(self.run_kernel()))).start()
        elif param_type == ML_MODEL:
            self.ml_model = self.ml_models[value]
            Thread(target=lambda: self.change_preview.emit(ML_READY,
                                                           self.nparray_to_qpixmap(self.run_ml()))).start()


    # def load_esrgan(self, device=None):
    #     if device is None:
    #         device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #     model_path = 'processing/models/ESRGAN/RRDB_ESRGAN_x4.pth'
    #     self.process_device = torch.device(device)
    #
    #     model = arch.RRDBNet(3, 3, 64, 23, gc=32)
    #     model.load_state_dict(torch.load(model_path), strict=True)
    #     model.eval()
    #     self.esrgan = model.to(self.process_device)

    # def esrgan_model(self, img):
    #     img = img * 1.0 / 255
    #     img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    #     lr_img = img.unsqueeze(0)
    #     lr_img = lr_img.to(self.process_device)
    #
    #     with torch.no_grad():
    #         torch.cuda.empty_cache()
    #         output = self.esrgan(lr_img).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    #     output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    #     output = np.array(output * 255.0, dtype=np.uint8)
    #     return output