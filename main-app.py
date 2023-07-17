import sys
from PyQt5.QtWidgets import QApplication
from loguru import logger

from utils import global_var as gl, logs
from utils.connect_mysql import db
from win.login_form import login_form
from win.splash.splash import SplashScreen

from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMenu, QAction
from main_win.win import Ui_mainWindow
from PyQt5.QtCore import Qt, QPoint, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QIcon

import json
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import os
import time
import cv2

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


from PyQt5 import QtWidgets, QtCore
from PyQt5.QtChart import QChart, QLineSeries, QChartView
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter
from PyQt5.QtWidgets import QMainWindow, QSystemTrayIcon, QFileSystemModel, QTreeView
from loguru import logger

from core.CpuLineChart import CpuLineChart
from core.DynamicSpline import DynamicSpline
from core.FileIconProvider import FileIconProvider
from core.ImageView import ImageView
from core.MetroCircleProgress import MetroCircleProgress
from core.MySystemTrayIcon import MySystemTrayIcon
from ui.main_window import Ui_MainWindow as main_window
from utils.CommonHelper import CommonHelper
from win.close_dialog import close_dialog
from utils import global_var as gl
from win.MainWindow import MainWindow

os.chdir(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class App(QApplication):
    def __init__(self):
        super().__init__(sys.argv)
        self.windows = {}

    def run(self, pytest=False):
        logger.info("程序启动 ...")

        splash = SplashScreen()  # 启动界面
        splash.loadProgress()  # 启动界面

        from win.main_win import main_win
        self.windows["main"] = main_win()
        self.windows["main-app"] = MainWindow()
        self.windows["login"] = login_form(self.windows["main-app"])
        self.windows["login"].show()

        # splash.finish(self.windows["main-app"])  # 启动界面

        if not pytest:
            sys.exit(self.exec_())

            # sys.exit(app.exec_())


if __name__ == "__main__":

    logs.setting()  # log 设置
    gl.__init()  # 全局变量
    db.connect()
    App().run()


    # app = QApplication(sys.argv)
    # # myWin = MainWindow()
    # # myWin.show()
    #
    # # myWin.showMaximized()
    # sys.exit(app.exec_())