import sys
from os import path, system, stat
from datetime import datetime

from PIL import Image
from PyQt5.QtCore import QEvent, pyqtSlot
from PyQt5.QtGui import QPainter, QPixmap
from PyQt5 import uic, QtCore, Qt, QtWidgets, QtGui
from PyQt5.QtOpenGL import QGLFormat
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QApplication, QAction, QGraphicsScene, QMessageBox, \
    QGraphicsView, QGraphicsTextItem, QDialog, QPushButton

from processing.ImageProcessing import ProcessingThread, INTERPOLATED_READY, KERNEL_READY, ML_READY, ML_MODEL, ESRGAN, \
    EDSR, FSRCNN, SCALE_FACTOR, INTER_METHOD, GAUSSIAN_BLUR, SHARPEN_FILTER, BILATERAL_FILTER, KERNEL_METHOD, \
    INTER_NEAREST, INTER_LINEAR, INTER_CUBIC, INTER_LANCZOS

import torch

PREVIEW_PIXELS_COUNT = 50


class SavingProgress(QDialog):

    def __init__(self, parent):
        super(SavingProgress, self).__init__(parent=parent)
        uic.loadUi('res/saving_progress.ui', self)
        self.setWindowFlags(Qt.Qt.Window | Qt.Qt.WindowTitleHint | Qt.Qt.CustomizeWindowHint)
        self.msg = QMessageBox()

        self.button_cancel.clicked.connect(self.cancel)
        self.button_open.clicked.connect(self.close)
        self.button_open.clicked.connect(lambda: self.setResult(1))

    def cancel(self):
        self.button_cancel: QPushButton
        if self.button_cancel.text() != 'Ok':
            msgbox = QMessageBox(QMessageBox.Question, 'Cancel?', 'Cancel image saving?')
            msgbox.addButton('No', QMessageBox.NoRole)
            msgbox.addButton('Yes', QMessageBox.YesRole)
            ok = msgbox.exec()
            if ok:
                self.close()
                self.setResult(-1)
        else:
            self.close()

    @pyqtSlot()
    def on_saved(self):
        try:
            self.button_cancel.setText('Ok')
            self.progressBar.setMaximum(100)
            self.progressBar.setValue(100)
            self.setWindowTitle('Saved')
            self.groupBox.setTitle('Image successfully saved!')
            self.button_open.setMaximumWidth(16777215)
        except Exception as e:
            print(e)


class SavingDialog(QDialog):

    def __init__(self, parent, img_path: str):
        super(SavingDialog, self).__init__(parent=parent)
        uic.loadUi('res/saving_dlg.ui', self)
        self.selected_method = ML_MODEL
        self.msg = QMessageBox()
        img_fmt = img_path[img_path.rindex('.'):]
        self.path = img_path[:img_path.rindex('.')] + datetime.now().strftime("%Y%m%d_%H%M%S") + img_fmt
        self.line_path.setText(self.path)
        self.button_path.clicked.connect(self.select_path)
        self.radio_inter.clicked.connect(lambda: self.method_changed(INTER_METHOD))
        self.radio_kernel.clicked.connect(lambda: self.method_changed(KERNEL_METHOD))
        self.radio_ml.clicked.connect(lambda: self.method_changed(ML_MODEL))

    def method_changed(self, method):
        self.selected_method = method

    def select_path(self):
        s_path = \
            QFileDialog.getSaveFileName(self, "Save image as", self.path, filter="Image(*.jpg *.png *.bmp *.jpeg)")[0]
        if s_path:
            self.path = s_path
            self.line_path.setText(self.path)

    def accept(self):
        if self.path:
            self.close()
            self.setResult(1)
        else:
            self.msg.setIcon(QMessageBox.Warning)
            self.msg.setWindowTitle('Err saving image')
            self.msg.setText('Select correct path for saving image file!')
            self.msg.exec_()


class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi('res/main_layout.ui', self)

        self.process_msg_params = {50: [3, 0.2, 0.4],
                                   100: [6, 0.25, 0.45],
                                   150: [9, 0.275, 0.475]}[PREVIEW_PIXELS_COUNT]
        self.checked_params = ['radio_scale_x4', 'radio_interp_nearest', 'radio_kernel_gaussian', 'radio_ml_frcnn']

        self.proc_th = ProcessingThread()
        self.zoom_factor = 0
        self.img_path = self.image = self.selected_frag = self.selection_scene = self.select_figure = None
        self.last_prev_pos, self.prev_pos = (0, 0), (1, 1)
        self.msg = QMessageBox()
        try:
            self.init_controls()
        except Exception as e:
            print(e)
        self.show()

    def init_controls(self):
        self.action_about = QAction("About app")
        self.menubar.addAction(self.action_about)
        self.action_open.triggered.connect(self.open_file)
        self.button_open.clicked.connect(self.open_file)
        self.action_save.triggered.connect(self.save_file)
        self.action_about.triggered.connect(lambda: self.send_message(
            QMessageBox.Information, "About:",
            "The application was created for educational purposes when performing a qualification paper on the topic "
            "\"Research of improvement methods quality images\""))
        self.proc_th.change_preview.connect(self.change_preview)

        self.selection_view.setCacheMode(self.selection_view.CacheBackground)
        self.selection_view.setRenderHints(
            QPainter.Antialiasing | QPainter.TextAntialiasing | QPainter.SmoothPixmapTransform)
        if QGLFormat.hasOpenGL():
            self.selection_view.setRenderHint(QPainter.HighQualityAntialiasing)
        self.selection_view.setViewportUpdateMode(self.selection_view.SmartViewportUpdate)
        self.selection_view.setHorizontalScrollBarPolicy(Qt.Qt.ScrollBarAlwaysOff)
        self.selection_view.setVerticalScrollBarPolicy(Qt.Qt.ScrollBarAlwaysOff)
        self.selection_view.setInteractive(True)
        self.selection_view.setDragMode(QGraphicsView.ScrollHandDrag)
        self.selection_view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.selection_view.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.selection_scene = QGraphicsScene()
        self.selection_view.setScene(self.selection_scene)
        self.selection_view.scene().installEventFilter(self)

        self.radio_interp_nearest.clicked.connect(
            lambda: self.preview_param_changed(INTER_METHOD, INTER_NEAREST, 'radio_interp_nearest'))
        self.radio_interp_linear.clicked.connect(
            lambda: self.preview_param_changed(INTER_METHOD, INTER_LINEAR, 'radio_interp_linear'))
        self.radio_interp_cubic.clicked.connect(
            lambda: self.preview_param_changed(INTER_METHOD, INTER_CUBIC, 'radio_interp_cubic'))
        self.radio_interp_lanczos.clicked.connect(
            lambda: self.preview_param_changed(INTER_METHOD, INTER_LANCZOS, 'radio_interp_lanczos'))

        self.radio_kernel_gaussian.clicked.connect(
            lambda: self.preview_param_changed(KERNEL_METHOD, GAUSSIAN_BLUR, 'radio_kernel_gaussian'))
        self.radio_kernel_sharpen.clicked.connect(
            lambda: self.preview_param_changed(KERNEL_METHOD, SHARPEN_FILTER, 'radio_kernel_sharpen'))
        self.radio_kernel_bilateral.clicked.connect(
            lambda: self.preview_param_changed(KERNEL_METHOD, BILATERAL_FILTER, 'radio_kernel_bilateral'))

        self.radio_ml_frcnn.clicked.connect(lambda: self.preview_param_changed(ML_MODEL, FSRCNN, 'radio_ml_frcnn'))
        self.radio_ml_edsr.clicked.connect(lambda: self.preview_param_changed(ML_MODEL, EDSR, 'radio_ml_edsr'))
        self.radio_ml_esrgan.clicked.connect(lambda: self.preview_param_changed(ML_MODEL, ESRGAN, 'radio_ml_esrgan'))

    def draw_image(self):
        if len(self.selection_view.scene().items()) == 2:
            self.selection_view.scene().clear()
            self.last_prev_pos, self.prev_pos = (0, 0), (1, 1)
        self.create_selection_fig()
        self.selection_scene.addPixmap(self.image)
        self.selection_view.scene().setSceneRect(0, 0, self.image.width(), self.image.height())

        self.selection_view.scene().addItem(self.select_figure)
        self.selection_view.fitInView(self.selection_view.scene().sceneRect(), Qt.Qt.KeepAspectRatio)
        self.update_pre()
        self.resize(self.geometry().width(), self.geometry().height())

    def eventFilter(self, obj, event) -> bool:
        if event.type() == QEvent.GraphicsSceneWheel:
            self.selection_zoom(event)
            event.accept()
            return True
        elif event.type() == QEvent.GraphicsSceneMouseRelease and len(self.selection_view.scene().items()) == 2:
            self.update_pre()
        elif event.type() == QEvent.MetaCall and len(self.selection_scene.items()) == 2:
            self.selection_fragment_changed()
        return False

    def selection_zoom(self, event):
        if event.delta() > 0:
            factor = 1.25
            self.zoom_factor += 1
        else:
            factor = 0.8
            self.zoom_factor -= 1
        if self.zoom_factor > 0:
            self.selection_view.scale(factor, factor)
        elif self.zoom_factor == 0:
            self.selection_view.fitInView(self.selection_view.scene().sceneRect(), Qt.Qt.KeepAspectRatio)
        else:
            self.zoom_factor = 0

    def selection_fragment_changed(self):
        x, y, w, h = self.select_figure.pos().x(), self.select_figure.pos().y(), \
                     self.image.width() - PREVIEW_PIXELS_COUNT, self.image.height() - PREVIEW_PIXELS_COUNT
        out_of_bounds = False
        if 0 > x:
            self.select_figure.setX(0)
            out_of_bounds = True
        elif x > w:
            self.select_figure.setX(w)
            out_of_bounds = True
        if 0 > y:
            self.select_figure.setY(0)
            out_of_bounds = True
        elif y > h:
            self.select_figure.setY(h)
            out_of_bounds = True
        if not out_of_bounds:
            self.last_prev_pos = (x, y)

    @pyqtSlot(int, QPixmap)
    def change_preview(self, pre_type, img):
        preview_obj = None
        if pre_type == INTERPOLATED_READY:
            preview_obj = self.inter_preview
        elif pre_type == KERNEL_READY:
            preview_obj = self.kernel_preview
        elif pre_type == ML_READY:
            preview_obj = self.ml_preview
        preview_obj.scene.clear()
        preview_obj.scene.addPixmap(img)
        preview_obj.fitInView(preview_obj.scene.itemsBoundingRect(), Qt.Qt.KeepAspectRatio)

    def show_upscale_form(self):
        self.group_orig.setMaximumSize(16777215, 16777215)
        self.group_inter.setMaximumSize(16777215, 16777215)
        self.group_kernel.setMaximumSize(16777215, 16777215)
        self.group_ml.setMaximumSize(16777215, 16777215)
        self.group_selection.setMaximumSize(16777215, 16777215)
        self.group_info.setMaximumSize(16777215, 16777215)
        self.group_params.setMaximumSize(16777215, 16777215)
        self.group_drop.setMaximumSize(0, 0)
        self.resize(self.geometry().width() + 1, self.geometry().height() + 1)

    def preview_param_changed(self, param_type, value, obj_name):
        if param_type == SCALE_FACTOR or param_type == INTER_METHOD:
            self.show_process_msg([[INTERPOLATED_READY, self.inter_preview]])
            if param_type == SCALE_FACTOR:
                self.checked_params[0] = obj_name
            else:
                self.checked_params[1] = obj_name
        elif param_type == KERNEL_METHOD:
            self.show_process_msg([[KERNEL_READY, self.kernel_preview]])
            self.checked_params[2] = obj_name
        elif param_type == ML_MODEL:
            self.show_process_msg([[ML_READY, self.ml_preview]])
            self.checked_params[3] = obj_name
        self.proc_th.change_param(param_type, value)

    def update_pre(self):
        if self.last_prev_pos != self.prev_pos:
            x, y = self.last_prev_pos
            self.selected_frag = self.selection_view.scene().items()[1].pixmap().copy(int(x), int(y),
                                                                                      PREVIEW_PIXELS_COUNT,
                                                                                      PREVIEW_PIXELS_COUNT)
            self.original_preview.scene = QGraphicsScene()
            self.original_preview.setScene(self.original_preview.scene)
            self.original_preview.scene.addPixmap(self.selected_frag)
            self.original_preview.fitInView(self.original_preview.scene.itemsBoundingRect(), Qt.Qt.KeepAspectRatio)

            self.show_process_msg()
            self.proc_th.set_pixmap(self.selected_frag)
            self.proc_th.start()

            self.prev_pos = self.last_prev_pos

    def show_process_msg(self, preview_fragments=None):
        if preview_fragments is None:
            preview_fragments = enumerate([self.inter_preview, self.kernel_preview, self.ml_preview])
        else:
            preview_fragments = preview_fragments
        io = QGraphicsTextItem()
        io.setPlainText("Processing...")
        font = io.font()
        p_size, coef_x, coef_y = self.process_msg_params
        font.setPointSize(p_size)
        io.setFont(font)
        io.setPos(PREVIEW_PIXELS_COUNT * coef_x, PREVIEW_PIXELS_COUNT * coef_y)
        msg_figure = QtWidgets.QGraphicsRectItem(QtCore.QRectF(PREVIEW_PIXELS_COUNT * 0.15,
                                                               PREVIEW_PIXELS_COUNT * 0.45,
                                                               PREVIEW_PIXELS_COUNT * 0.7,
                                                               PREVIEW_PIXELS_COUNT * 0.2))
        msg_figure.setBrush(Qt.Qt.white)
        try:
            for preview_type, preview_obj in preview_fragments:
                preview_obj.scene = QGraphicsScene()
                preview_obj.setScene(preview_obj.scene)
                self.change_preview(preview_type, self.selected_frag)
                preview_obj.scene.addItem(msg_figure)
                preview_obj.scene.addItem(io)
        except Exception as e:
            print(e)

    def dropEvent(self, event):
        file_name = [u.toLocalFile() for u in event.mimeData().urls()][0]
        if path.isfile(file_name):
            self.open_file(file_name)
            self.activateWindow()
        else:
            print('Can\'t open dir')

    def open_file(self, img_path=None):
        if not img_path:
            img_path = QFileDialog.getOpenFileName(self, "Choose an Image", filter="Image(*.jpg *.png *.bmp *.jpeg)")[0]
        if not img_path:
            return
        if self.group_orig.maximumWidth() == 0:
            self.show_upscale_form()
        self.img_path = img_path
        self.image = QPixmap(img_path)
        self.load_image_info()
        self.draw_image()

    def save_file(self):
        try:
            s_dlg = SavingDialog(self, self.img_path)
            ok = s_dlg.exec_()
            save_img_path, method = s_dlg.path, s_dlg.selected_method
            if ok:
                s_p = SavingProgress(self)

                @pyqtSlot(QPixmap)
                def save_image(image):
                    image.save(save_img_path)
                    s_p.on_saved()

                self.proc_th.init_full_saving(method, self.image)
                self.proc_th.image_saved.connect(save_image)
                self.proc_th.start()
                result = s_p.exec_()
                if result == 1:
                    system("start " + self.img_path.replace('\\', '/'))
                    system("start " + save_img_path.replace('\\', '/'))
                elif result == -1:
                    self.proc_th.terminate()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                self.proc_th.image_saved.disconnect()

        except Exception as e:
            print(e)

    def load_image_info(self):
        self.setWindowTitle('Image Enhancer – ' + self.img_path)
        image = Image.open(self.img_path)
        size = stat(self.img_path).st_size
        for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024.0:
                size = "%3.1f %s" % (size, x)
                break
            size /= 1024.0
        self.label_img_fmt.setText(image.format)
        self.label_img_size.setText(size)
        self.label_img_width.setText(str(image.width) + 'px')
        self.label_img_height.setText(str(image.height) + 'px')
        self.label_img_color.setText(image.mode)

    def create_selection_fig(self):
        self.select_figure = QtWidgets.QGraphicsRectItem(QtCore.QRectF(0, 0,
                                                                       PREVIEW_PIXELS_COUNT,
                                                                       PREVIEW_PIXELS_COUNT))
        self.select_figure.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, True)
        self.select_figure.setBrush(Qt.Qt.white)
        self.select_figure.setOpacity(0.65)
        p = self.select_figure.pen()
        p.setWidth(3)
        self.select_figure.setPen(p)

    def send_message(self, type_msg, title_msg, text_msg):
        self.msg.setIcon(type_msg)
        self.msg.setWindowTitle(title_msg)
        self.msg.setText(text_msg)
        self.msg.exec_()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        self.selection_view.fitInView(self.selection_view.scene().itemsBoundingRect(), Qt.Qt.KeepAspectRatio)
        if self.image:
            self.original_preview.fitInView(self.original_preview.scene.itemsBoundingRect(), Qt.Qt.KeepAspectRatio)
            self.inter_preview.fitInView(self.inter_preview.scene.itemsBoundingRect(), Qt.Qt.KeepAspectRatio)
            self.kernel_preview.fitInView(self.kernel_preview.scene.itemsBoundingRect(), Qt.Qt.KeepAspectRatio)
            self.ml_preview.fitInView(self.ml_preview.scene.itemsBoundingRect(), Qt.Qt.KeepAspectRatio)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec_())
    # test()
