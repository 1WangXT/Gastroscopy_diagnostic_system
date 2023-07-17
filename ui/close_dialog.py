# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui/close_dialog.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(377, 272)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/icon/icon/关闭.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        Dialog.setWindowIcon(icon)
        Dialog.setStyleSheet("QTextEdit#textEdit{\n"
" background-color: rgba(0,0,0,0);\n"
" selection-background-color:#88bbff;\n"
" border: 0px solid rgba(0,0,0,0);\n"
"}\n"
"\n"
"QLabel#label{\n"
" color: rgba(0,0,0,0);\n"
"}\n"
"\n"
"QWidget#widget {\n"
" border-image: url(:/img/image/load.png);\n"
" border-radius:10px;\n"
" background-color: rgba(0,0,0,0);\n"
"}\n"
"\n"
"QDialogButtonBox [text=\"OK\"] {\n"
" qproperty-text: \"好的\";\n"
"}\n"
"QDialogButtonBox [text=\"Save\"] {\n"
" qproperty-text: \"保存\";\n"
"}\n"
"QDialogButtonBox [text=\"Save All\"] {\n"
" qproperty-text: \"保存全部\";\n"
"}\n"
"\n"
"QPushButton#close_pushButton{\n"
" background-color: #ce5137;\n"
" border-radius:10px;\n"
"}\n"
"\n"
"QPushButton#close_pushButton:hover{\n"
" background-size: cover;\n"
" background-image: url(:/icon/icon/close.svg);\n"
"}\n"
"\n"
"QPushButton#min_pushButton{\n"
" background-color: #a1c661;\n"
" border-radius:10px;\n"
"}\n"
"\n"
"QPushButton#min_pushButton:hover{\n"
" background-size: cover;\n"
" background-image: url(:/icon/icon/minimize.svg);\n"
"}")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(Dialog)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.widget = QtWidgets.QWidget(Dialog)
        self.widget.setObjectName("widget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.title_label = QtWidgets.QLabel(self.widget)
        font = QtGui.QFont()
        font.setFamily("华文楷体")
        font.setBold(True)
        font.setWeight(75)
        self.title_label.setFont(font)
        self.title_label.setObjectName("title_label")
        self.horizontalLayout_2.addWidget(self.title_label)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.min_pushButton = QtWidgets.QPushButton(self.widget)
        self.min_pushButton.setMinimumSize(QtCore.QSize(20, 20))
        self.min_pushButton.setMaximumSize(QtCore.QSize(20, 20))
        self.min_pushButton.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.min_pushButton.setText("")
        self.min_pushButton.setObjectName("min_pushButton")
        self.horizontalLayout_2.addWidget(self.min_pushButton)
        self.close_pushButton = QtWidgets.QPushButton(self.widget)
        self.close_pushButton.setMinimumSize(QtCore.QSize(20, 20))
        self.close_pushButton.setMaximumSize(QtCore.QSize(20, 20))
        self.close_pushButton.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.close_pushButton.setText("")
        self.close_pushButton.setObjectName("close_pushButton")
        self.horizontalLayout_2.addWidget(self.close_pushButton)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.add_widget = QtWidgets.QWidget(self.widget)
        self.add_widget.setObjectName("add_widget")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.add_widget)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.is_min_status_checkBox = QtWidgets.QCheckBox(self.add_widget)
        self.is_min_status_checkBox.setChecked(True)
        self.is_min_status_checkBox.setObjectName("is_min_status_checkBox")
        self.verticalLayout_4.addWidget(self.is_min_status_checkBox)
        self.verticalLayout.addWidget(self.add_widget)
        self.textEdit = QtWidgets.QTextEdit(self.widget)
        self.textEdit.setObjectName("textEdit")
        self.verticalLayout.addWidget(self.textEdit)
        self.buttonBox = QtWidgets.QDialogButtonBox(self.widget)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.buttonBox.setFont(font)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setCenterButtons(False)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout.addWidget(self.buttonBox)
        self.verticalLayout_3.addWidget(self.widget)

        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept)
        self.buttonBox.rejected.connect(Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.title_label.setText(_translate("Dialog", "<html><head/><body><p><span style=\" font-size:11pt; color:#ffaa7f;\">关闭程序</span></p></body></html>"))
        self.is_min_status_checkBox.setText(_translate("Dialog", "最小化到任务栏图标"))
from res import app_rc