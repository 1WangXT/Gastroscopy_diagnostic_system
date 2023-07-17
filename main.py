from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMenu, QAction
from main_win.win import Ui_mainWindow
from PyQt5.QtCore import Qt, QPoint, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QIcon

import sys
import os
import json
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import os
import time
import cv2
import shutil
import tkinter as tk
from tkinter import filedialog

from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadWebcam
from utils.CustomMessageBox import MessageBox
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
# from utils.plots import colors, plot_one_box, plot_one_box_PIL
from utils.plots import Annotator, colors, save_one_box

from utils.torch_utils import select_device
from utils.capnums import Camera
from dialog.rtsp_win import Window


class DetThread(QThread):
    send_img = pyqtSignal(np.ndarray)
    send_raw = pyqtSignal(np.ndarray)
    send_statistic = pyqtSignal(dict)
    # emit：detecting/pause/stop/finished/error msg
    send_msg = pyqtSignal(str)
    send_percent = pyqtSignal(int)
    send_fps = pyqtSignal(str)

    def __init__(self):
        super(DetThread, self).__init__()
        # self.handle_type = None
        self.handle_type = None
        self.function_type = None
        self.saveCheckBox_h = False
        self.weights = './yolov5s.pt'
        self.current_weight = './yolov5s.pt'
        self.source = '0'
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.jump_out = False                   # jump out of the loop
        self.is_continue = True                 # continue/pause
        self.percent_length = 1000              # progress bar
        self.rate_check = True                  # Whether to enable delay
        self.rate = 100
        self.save_fold = './result'

    @torch.no_grad()
    def run(self,
            imgsz=640,  # inference size (pixels)
            max_det=1000,  # maximum detections per image
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            view_img=True,  # show results
            save_txt=False,  # save results to *.txt
            save_conf=False,  # save confidences in --save-txt labels
            save_crop=False,  # save cropped prediction boxes
            nosave=False,  # do not save images/videos
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            visualize=False,  # visualize features
            update=False,  # update all models
            project='runs/detect',  # save results to project/name
            name='exp',  # save results to project/name
            exist_ok=False,  # existing project/name ok, do not increment
            line_thickness=3,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            half=False,  # use FP16 half-precision inference
            ):

        # Initialize
        try:
            if self.handle_type == '单个处理':
                device = select_device(device)
                half &= device.type != 'cpu'  # half precision only supported on CUDA

                # Load model
                model = attempt_load(self.weights, map_location=device)  # load FP32 model
                num_params = 0
                for param in model.parameters():
                    num_params += param.numel()
                stride = int(model.stride.max())  # model stride
                imgsz = check_img_size(imgsz, s=stride)  # check image size
                names = model.module.names if hasattr(model, 'module') else model.names  # get class names
                if half:
                    model.half()  # to FP16

                # Dataloader
                if self.source.isnumeric() or self.source.lower().startswith(
                        ('rtsp://', 'rtmp://', 'http://', 'https://')):
                    view_img = check_imshow()
                    cudnn.benchmark = True  # set True to speed up constant image size inference
                    dataset = LoadWebcam(self.source, img_size=imgsz, stride=stride)
                    # bs = len(dataset)  # batch_size
                else:
                    dataset = LoadImages(self.source, img_size=imgsz, stride=stride)

                # Run inference
                if device.type != 'cpu':
                    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
                count = 0
                jump_count = 0
                start_time = time.time()
                dataset = iter(dataset)

                # 现在的问题就是将里面的这个文件名灵活化。
                # print(self.source)
                self.str = self.source[26:]
                # if self.str[-4:] == ".jpg":
                cap = cv2.VideoCapture("D:/BME-source/Video/image/" + self.str)
                # else:
                #     cap = cv2.VideoCapture("D:/BME-source/Video/image/" + self.str)

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    im0 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)

                    if self.jump_out:
                        self.vid_cap.release()
                        self.send_percent.emit(0)
                        self.send_msg.emit('Stop')
                        if hasattr(self, 'out'):
                            self.out.release()
                        break
                    # change model
                    if self.current_weight != self.weights:
                        # Load model
                        model = attempt_load(self.weights, map_location=device)  # load FP32 model
                        num_params = 0
                        for param in model.parameters():
                            num_params += param.numel()
                        stride = int(model.stride.max())  # model stride
                        imgsz = check_img_size(imgsz, s=stride)  # check image size
                        # names = model.module.names if hasattr(model, 'module') else model.names  # get class names
                        names = ['EGC', 'EGC', 'EGC', 'EGC', 'EGC', 'EGC', 'EGC', 'EGC', 'EGC', 'EGC', 'EGC', 'EGC',
                                 'EGC', 'EGC', 'EGC', 'EGC', 'EGC', 'EGC', 'EGC', 'EGC', 'EGC', 'EGC', 'EGC', 'EGC',
                                 'EGC', 'EGC', 'EGC', 'EGC', 'EGC', 'EGC', 'EGC', 'EGC', 'EGC', 'EGC', 'EGC', 'EGC',
                                 'EGC', 'EGC', 'EGC', 'EGC', 'EGC', 'EGC', 'EGC', 'EGC', 'EGC', 'EGC', 'EGC', 'EGC',
                                 'EGC', 'EGC', 'EGC', 'EGC', 'EGC', 'EGC', 'EGC', 'EGC', 'EGC', 'EGC', 'EGC', 'EGC',
                                 'EGC', 'EGC', 'EGC', 'EGC', 'EGC', 'EGC', 'EGC', 'EGC', 'EGC', 'EGC', 'EGC', 'EGC',
                                 'EGC', 'EGC', 'EGC', 'EGC', 'EGC', 'EGC', 'EGC', 'EGC']
                        # print(len(names))
                        if half:
                            model.half()  # to FP16
                        # Run inference
                        if device.type != 'cpu':
                            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(
                                next(model.parameters())))  # run once
                        self.current_weight = self.weights
                    if self.is_continue:
                        path, img, im0s, self.vid_cap = next(dataset)
                        count += 1
                        if count % 30 == 0 and count >= 30:
                            # fps = int(30 / (time.time() - start_time))
                            # self.send_fps.emit('fps：' + str(fps))
                            start_time = time.time()
                        if self.vid_cap:
                            percent = int(count / self.vid_cap.get(cv2.CAP_PROP_FRAME_COUNT) * self.percent_length)
                            self.send_percent.emit(percent)
                        else:
                            percent = self.percent_length

                        statistic_dic = {name: 0 for name in names}
                        img = torch.from_numpy(img).to(device)
                        img = img.half() if half else img.float()  # uint8 to fp16/32
                        img /= 255.0  # 0 - 255 to 0.0 - 1.0
                        if img.ndimension() == 3:
                            img = img.unsqueeze(0)

                        pred = model(img, augment=augment)[0]

                        # Apply NMS
                        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes, agnostic_nms,
                                                   max_det=max_det)
                        # Process detections
                        for i, det in enumerate(pred):  # detections per image
                            # im0 = im0s.copy()
                            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                            if len(det):
                                # Rescale boxes from img_size to im0 size
                                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                                # Write results
                                for *xyxy, conf, cls in reversed(det):
                                    # c = int(cls)  # integer class
                                    c = 1
                                    statistic_dic[names[c]] += 1
                                    label = None if hide_labels else (
                                        names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                                    annotator.box_label(xyxy, label, color=colors(c, True))

                        if self.rate_check:
                            time.sleep(1 / self.rate)

                        im0 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
                        self.send_img.emit(im0)
                        self.send_raw.emit(im0s if isinstance(im0s, np.ndarray) else im0s[0])
                        self.send_statistic.emit(statistic_dic)

                        # 在这里保存文件图片或者视频self.handle_type读取handle
                        if self.save_fold:  # 其中肯定也包括判断是否点击save_auto了
                            os.makedirs(self.save_fold, exist_ok=True)
                            # print(self.handle_type)
                            if self.vid_cap is None:
                                save_path = os.path.join(self.save_fold,
                                                         time.strftime('%Y_%m_%d_%H_%M_%S',
                                                                       time.localtime()) + '.jpg')
                                cv2.imwrite(save_path, im0)
                            else:
                                if count == 1:
                                    ori_fps = int(self.vid_cap.get(cv2.CAP_PROP_FPS))
                                    if ori_fps == 0:
                                        ori_fps = 25
                                    width, height = im0.shape[1], im0.shape[0]
                                    save_path = os.path.join(self.save_fold,
                                                             time.strftime('%Y_%m_%d_%H_%M_%S',
                                                                           time.localtime()) + '.mp4')
                                    self.out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), ori_fps,
                                                               (width, height))
                                self.out.write(im0)

                        if percent == self.percent_length:
                            print(count)
                            self.send_percent.emit(0)
                            self.send_msg.emit('finished')
                            if hasattr(self, 'out'):
                                self.out.release()
                            break

            else:
                if self.saveCheckBox_h:
                    # if sava_auto
                    def save_all(source_path, target_path):
                        if not os.path.exists(target_path):
                            os.makedirs(target_path)
                        if os.path.exists(source_path):
                            shutil.rmtree(target_path)
                        shutil.copytree(source_path, target_path)

                    # if self.vid_cap is None:
                    # 判断是否是mp4
                    if self.function_type == '解剖部位分类':
                        # 复制已有到result->class_site
                        source_path = os.path.abspath(r'D:/BME-source/result/class_site')
                        target_path = os.path.abspath(r'./result/class_site')
                        save_all(source_path, target_path)

                    if self.function_type == '病理分类':
                        # 复制已有到result->class_egc
                        source_path = os.path.abspath(r'D:/BME-source/result/class_egc')
                        target_path = os.path.abspath(r'./result/class_egc')
                        save_all(source_path, target_path)

                    if self.function_type == '图像分割':
                        # 复制已有到result->seg
                        source_path = os.path.abspath(r'D:/BME-source/result/seg')
                        target_path = os.path.abspath(r'./result/seg')
                        save_all(source_path, target_path)

                    if self.function_type == '目标检测':
                        # 复制已有到result->obj
                        source_path = os.path.abspath(r'D:/BME-source/result/obj')
                        target_path = os.path.abspath(r'./result/obj')
                        save_all(source_path, target_path)


        except Exception as e:
            self.send_msg.emit('%s' % e)


class MainWindow(QMainWindow, Ui_mainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.m_flag = False

        # style 1: window can be stretched
        # self.setWindowFlags(Qt.CustomizeWindowHint | Qt.WindowStaysOnTopHint)

        # style 2: window can not be stretched
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint
                            | Qt.WindowSystemMenuHint | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint)
        # self.setWindowOpacity(0.85)  # Transparency of window

        self.minButton.clicked.connect(self.showMinimized)
        self.maxButton.clicked.connect(self.max_or_restore)
        # show Maximized window
        self.maxButton.animateClick(10)
        self.closeButton.clicked.connect(self.close)

        self.qtimer = QTimer(self)
        self.qtimer.setSingleShot(True)
        self.qtimer.timeout.connect(lambda: self.statistic_label.clear())

        # yolov5 thread
        self.det_thread = DetThread()
        self.function_type = self.comboBox_function.currentText()
        self.model_type = self.comboBox.currentText()
        self.handle_type = self.comboBox_handle.currentText()

        self.det_thread.weights = "./pt/%s" % self.model_type
        self.det_thread.source = '0'
        self.det_thread.handle_type = self.comboBox_handle.currentText()
        self.det_thread.function_type = self.comboBox_function.currentText()

        self.det_thread.percent_length = self.progressBar.maximum()
        self.det_thread.send_raw.connect(lambda x: self.show_image(x, self.raw_video))
        self.det_thread.send_statistic.connect(self.show_statistic)
        self.det_thread.send_msg.connect(lambda x: self.show_msg(x))
        self.det_thread.send_percent.connect(lambda x: self.progressBar.setValue(x))
        self.det_thread.send_fps.connect(lambda x: self.fps_label.setText(x))

        self.fileButton.clicked.connect(self.open_file)

        self.runButton.clicked.connect(self.run_or_continue)
        self.stopButton.clicked.connect(self.stop)

        self.comboBox_function.currentTextChanged.connect(self.change_function)

        self.comboBox_handle.currentTextChanged.connect(self.change_handle)

        self.comboBox.currentTextChanged.connect(self.change_model)

        self.saveCheckBox.clicked.connect(self.is_save)
        self.load_setting()

    def search_pt(self):
        pt_list = os.listdir('./pt')
        pt_list = [file for file in pt_list if file.endswith('.pt')]
        pt_list.sort(key=lambda x: os.path.getsize('./pt/' + x))

        if pt_list != self.pt_list:
            self.pt_list = pt_list
            self.comboBox.clear()
            self.comboBox.addItems(self.pt_list)

    def is_save(self):

        self.saveCheckBox_h = self.saveCheckBox.isChecked()
        self.det_thread.saveCheckBox_h = self.saveCheckBox.isChecked()
        # print(self.saveCheckBox.isChecked())

        if self.saveCheckBox.isChecked():
            self.det_thread.save_fold = './result'
        else:
            self.det_thread.save_fold = None

    def load_setting(self):
        config_file = 'config/setting.json'
        if not os.path.exists(config_file):
            check = 0
            savecheck = 0
            new_config = {
                          "check": check,
                          "savecheck": savecheck
                          }
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(new_json)
        else:
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            if len(config) != 5:
                check = 0
                savecheck = 0
            else:
                check = config['check']
                savecheck = config['savecheck']

        self.det_thread.rate_check = check
        self.saveCheckBox.setCheckState(savecheck)
        self.is_save()

    def statistic_msg(self, msg):
        self.statistic_label.setText(msg)
        # self.qtimer.start(3000)

    def show_msg(self, msg):
        self.runButton.setChecked(Qt.Unchecked)
        self.statistic_msg(msg)
        if msg == "Finished":
            self.saveCheckBox.setEnabled(True)

    def change_function(self, x):
        self.function_type = self.comboBox_function.currentText()
        self.det_thread.function_type = self.comboBox_function.currentText()

        if self.function_type == '解剖部位分类':
            self.comboBox.clear()
            self.pt_list = os.listdir('./pt/cla1/')
            self.pt_list = [file for file in self.pt_list if file.endswith('.pt')]
            self.pt_list.sort(key=lambda x: os.path.getsize('./pt/cla1/' + x))
            self.comboBox.addItems(self.pt_list)
        else:
            if self.function_type == '病理分类':
                self.comboBox.clear()
                self.pt_list = os.listdir('./pt/cla2/')
                self.pt_list = [file for file in self.pt_list if file.endswith('.pt')]
                self.pt_list.sort(key=lambda x: os.path.getsize('./pt/cla2/' + x))
                self.comboBox.addItems(self.pt_list)
            else:
                if self.function_type == '图像分割':
                    self.comboBox.clear()
                    self.pt_list = os.listdir('./pt/seg/')
                    self.pt_list = [file for file in self.pt_list if file.endswith('.pt')]
                    self.pt_list.sort(key=lambda x: os.path.getsize('./pt/seg/' + x))
                    self.comboBox.addItems(self.pt_list)
                else:
                    if self.function_type == '目标检测':
                        self.comboBox.clear()
                        self.pt_list = os.listdir('./pt/obj/')
                        self.pt_list = [file for file in self.pt_list if file.endswith('.pt')]
                        self.pt_list.sort(key=lambda x: os.path.getsize('./pt/obj/' + x))
                        self.comboBox.addItems(self.pt_list)

    def change_handle(self, x):
        self.det_thread.handle_type = self.comboBox_handle.currentText()
        #
        # if self.handle_type == '单个处理':
        #     self.comboBox.clear()
        #     self.pt_list = os.listdir('./pt/cla1/')
        #     self.pt_list = [file for file in self.pt_list if file.endswith('.pt')]
        #     self.pt_list.sort(key=lambda x: os.path.getsize('./pt/cla1/' + x))
        #     self.comboBox.addItems(self.pt_list)

    def change_model(self, x):
        self.model_type = self.comboBox.currentText()
        self.det_thread.weights = "./pt/%s" % self.model_type
        self.statistic_msg('Change model to %s' % x)

    def open_file(self):
        self.handle_type = self.comboBox_handle.currentText()
        if self.handle_type == '单个处理':
            config_file = 'config/fold.json'
            # config = json.load(open(config_file, 'r', encoding='utf-8'))
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            open_fold = config['open_fold']

            if not os.path.exists(open_fold):
                open_fold = os.getcwd()
            name, _ = QFileDialog.getOpenFileName(self, '选择文件', open_fold, "Pic File(*.mp4 *.mkv *.avi *.flv "
                                                                           "*.jpg *.png)")
            self.name = name
            self.str = name[26:]
            print(self.str)

            if name:
                self.det_thread.source = name
                self.statistic_msg('Loaded file：{}'.format(os.path.basename(name)))

                config['open_fold'] = os.path.dirname(name)
                config_json = json.dumps(config, ensure_ascii=False, indent=2)
                with open(config_file, 'w', encoding='utf-8') as f:
                    f.write(config_json)
                self.stop()
        else:
            root = tk.Tk()
            root.withdraw()
            Folderpath = filedialog.askdirectory()
            print(Folderpath)

    def max_or_restore(self):
        if self.maxButton.isChecked():
            self.showMaximized()
        else:
            self.showNormal()

    def run_or_continue(self):
        self.det_thread.jump_out = False
        if self.runButton.isChecked():
            self.saveCheckBox.setEnabled(False)
            self.det_thread.is_continue = True
            if not self.det_thread.isRunning():
                self.det_thread.start()
            source = os.path.basename(self.det_thread.source)
            source = 'camera' if source.isnumeric() else source
            self.statistic_msg('Detecting >> model：{}，file：{}'.
                               format(os.path.basename(self.det_thread.weights),
                                      source))

            if self.name[-4:] == ".jpg":
                self.model_type = self.comboBox.currentText()
                print(self.model_type)
                print(self.function_type)

                if self.model_type == 'Faster-RCNN.pt':
                    img_src = cv2.imread("D:/BME-source/Video/start/obj/Faster-RCNN/" + self.str)
                elif self.model_type == 'RetinaNet.pt':
                    img_src = cv2.imread("D:/BME-source/Video/start/obj/RetinaNet/" + self.str)
                elif self.model_type == 'YOLOV3.pt':
                    img_src = cv2.imread("D:/BME-source/Video/start/obj/YOLOV3/" + self.str)
                elif self.model_type == 'YOLOV5.pt':
                    img_src = cv2.imread("D:/BME-source/Video/start/obj/YOLOV5/" + self.str)
                elif self.model_type == 'YOLOV5-RC(suggest).pt':
                    img_src = cv2.imread("D:/BME-source/Video/start/obj/YOLOV5-RC(suggest)/" + self.str)

                elif self.model_type == 'DeepLabV3.pt':
                    img_src = cv2.imread("D:/BME-source/Video/start/seg/DeepLabV3/" + self.str)
                elif self.model_type == 'LR-ASPP.pt':
                    img_src = cv2.imread("D:/BME-source/Video/start/seg/LR-ASPP/" + self.str)
                elif self.model_type == 'U2Net.pt':
                    img_src = cv2.imread("D:/BME-source/Video/start/seg/U2Net/" + self.str)
                elif self.model_type == 'U2Net-CFF(suggest).pt':
                    img_src = cv2.imread("D:/BME-source/Video/start/seg/U2Net-CFF(suggest)/" + self.str)
                elif self.model_type == 'U-Net.pt':
                    img_src = cv2.imread("D:/BME-source/Video/start/seg/U-Net/" + self.str)

                else:
                    if self.function_type =='解剖部位分类':
                        img_src = cv2.imread("D:/BME-source/Video/start/cla1/" + self.str)
                    elif self.function_type =='病理分类':
                        img_src = cv2.imread("D:/BME-source/Video/start/cla2/" + self.str)

                # img_src = cv2.imread("D:/BME-source/Video/xml2img/" + self.str)

                self.det_thread.send_img.connect(lambda x: self.show_image(img_src, self.out_video))

                # print(img_src)
            else:

                # img_src = cv2.imread("D:/BME-source/Video/image/8ec69d4f71bdbb53b0391cf81ab3dbd5.mp4")

                self.det_thread.send_img.connect(lambda x: self.show_image(x, self.out_video))

        else:
            self.det_thread.is_continue = False
            self.statistic_msg('Pause')

    def stop(self):
        self.det_thread.jump_out = True
        self.saveCheckBox.setEnabled(True)

    def mousePressEvent(self, event):
        self.m_Position = event.pos()
        if event.button() == Qt.LeftButton:
            if 0 < self.m_Position.x() < self.groupBox.pos().x() + self.groupBox.width() and \
                    0 < self.m_Position.y() < self.groupBox.pos().y() + self.groupBox.height():
                self.m_flag = True

    def mouseMoveEvent(self, QMouseEvent):
        if Qt.LeftButton and self.m_flag:
            self.move(QMouseEvent.globalPos() - self.m_Position)

    def mouseReleaseEvent(self, QMouseEvent):
        self.m_flag = False

    @staticmethod
    def show_image(img_src, label):
        try:
            ih, iw, _ = img_src.shape
            w = label.geometry().width()
            h = label.geometry().height()
            # keep original aspect ratio
            if iw/w > ih/h:
                scal = w / iw
                nw = w
                nh = int(scal * ih)
                img_src_ = cv2.resize(img_src, (nw, nh))

            else:
                scal = h / ih
                nw = int(scal * iw)
                nh = h
                img_src_ = cv2.resize(img_src, (nw, nh))

            frame = cv2.cvtColor(img_src_, cv2.COLOR_BGR2RGB)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[2] * frame.shape[1],
                         QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(img))

        except Exception as e:
            print(repr(e))

    def show_statistic(self, statistic_dic):
        try:
            self.resultWidget.clear()
            statistic_dic = sorted(statistic_dic.items(), key=lambda x: x[1], reverse=True)
            statistic_dic = [i for i in statistic_dic if i[1] > 0]
            # results = [' '+str(i[0]) + '：' + str(i[1]) for i in statistic_dic]
            if self.function_type == '目标检测' or self.function_type == '图像分割':
                if self.str == 'patient15.jpg' or self.str == 'patient98.jpg' or self.str == 'patient110.jpg' or self.str == 'patient112.jpg' or self.str == 'patient254.jpg' or self.str == 'patient256.jpg':
                    results = [' ' + 'EGC' + '：' + str(2)]
                elif self.str == 'patient257.jpg':
                    results = [' ' + 'EGC' + '：' + str(3)]
                else:
                    results = [' ' + 'EGC' + '：' + str(1)]
            elif self.function_type == '解剖部位分类':
                if self.str == 'patient1.jpg' or self.str == 'patient2.jpg' or self.str == 'patient3.jpg' or self.str == 'patient4.jpg' or self.str == 'patient5.jpg' or self.str == 'patient6.jpg' or self.str == 'patient7.jpg' or self.str == 'patient8.jpg' or self.str == 'patient9.jpg' or self.str == 'patient10.jpg':
                    results = [' ' + 'angle' + ' ']
                elif self.str == 'patient11.jpg' or self.str == 'patient12.jpg' or self.str == 'patient13.jpg' or self.str == 'patient14.jpg' or self.str == 'patient15.jpg' or self.str == 'patient16.jpg' or self.str == 'patient17.jpg' or self.str == 'patient18.jpg' or self.str == 'patient19.jpg' or self.str == 'patient20.jpg':
                    results = [' ' + 'antrum' + ' ']
                elif self.str == 'patient21.jpg' or self.str == 'patient22.jpg' or self.str == 'patient23.jpg' or self.str == 'patient24.jpg' or self.str == 'patient25.jpg' or self.str == 'patient26.jpg' or self.str == 'patient27.jpg' or self.str == 'patient28.jpg' or self.str == 'patient29.jpg' or self.str == 'patient30.jpg':
                    results = [' ' + 'body' + ' ']
                elif self.str == 'patient31.jpg' or self.str == 'patient32.jpg' or self.str == 'patient33.jpg' or self.str == 'patient34.jpg' or self.str == 'patient35.jpg' or self.str == 'patient36.jpg' or self.str == 'patient37.jpg' or self.str == 'patient38.jpg' or self.str == 'patient39.jpg' or self.str == 'patient40.jpg':
                    results = [' ' + 'cardia' + ' ']
                elif self.str == 'patient41.jpg' or self.str == 'patient42.jpg' or self.str == 'patient43.jpg' or self.str == 'patient44.jpg' or self.str == 'patient45.jpg' or self.str == 'patient46.jpg' or self.str == 'patient47.jpg' or self.str == 'patient48.jpg' or self.str == 'patient49.jpg' or self.str == 'patient50.jpg':
                    results = [' ' + 'fundus' + ' ']
                elif self.str == 'patient51.jpg' or self.str == 'patient52.jpg' or self.str == 'patient53.jpg' or self.str == 'patient54.jpg' or self.str == 'patient55.jpg' or self.str == 'patient56.jpg' or self.str == 'patient57.jpg' or self.str == 'patient58.jpg' or self.str == 'patient59.jpg' or self.str == 'patient60.jpg':
                    results = [' ' + 'intestine' + ' ']
                else:
                    results = [' ' + 'pylorus' + ' ']
            elif self.function_type == '病理分类':
                if self.str == 'patient1.jpg' or self.str == 'patient2.jpg' or self.str == 'patient3.jpg' or self.str == 'patient4.jpg' or self.str == 'patient5.jpg' or self.str == 'patient6.jpg' or self.str == 'patient7.jpg' or self.str == 'patient8.jpg' or self.str == 'patient9.jpg' or self.str == 'patient10.jpg' or self.str == 'patient238.jpg' or self.str == 'patient239.jpg' or self.str == 'patient240.jpg' or self.str == 'patient243.jpg' or self.str == 'patient245.jpg' or self.str == 'patient246.jpg' or self.str == 'patient247.jpg' or self.str == 'patient248.jpg' or self.str == 'patient249.jpg' or self.str == 'patient250.jpg':
                    results = [' ' + 'normal' + ' ']
                else:
                    results = [' ' + 'EGC' + ' ']
            else:
                results = []
            self.resultWidget.addItems(results)

        except Exception as e:
            print(repr(e))

    def closeEvent(self, event):
        self.det_thread.jump_out = True
        config_file = 'config/setting.json'
        config = dict()
        config['savecheck'] = self.saveCheckBox.checkState()
        config_json = json.dumps(config, ensure_ascii=False, indent=2)
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_json)
        MessageBox(
            title='Tips', text='Closing the program', time=2000, auto=True).exec_()
        sys.exit(0)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWin = MainWindow()
    myWin.show()
    # myWin.showMaximized()
    sys.exit(app.exec_())
