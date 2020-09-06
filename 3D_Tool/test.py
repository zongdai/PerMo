# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'test.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets
from widgets import GraphicsView

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1920, 1080)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.filesList = QtWidgets.QListView(self.centralwidget)
        self.filesList.setGeometry(QtCore.QRect(1610, 350, 291, 341))
        self.filesList.setObjectName("filesList")
        self.x_slider = QtWidgets.QSlider(self.centralwidget)
        self.x_slider.setGeometry(QtCore.QRect(40, 110, 211, 31))
        self.x_slider.setMinimum(-3000)
        self.x_slider.setMaximum(3000)
        self.x_slider.setOrientation(QtCore.Qt.Horizontal)
        self.x_slider.setObjectName("x_slider")
        self.y_slider = QtWidgets.QSlider(self.centralwidget)
        self.y_slider.setGeometry(QtCore.QRect(40, 180, 211, 31))
        self.y_slider.setMinimum(-200)
        self.y_slider.setMaximum(700)
        self.y_slider.setSliderPosition(30)
        self.y_slider.setOrientation(QtCore.Qt.Horizontal)
        self.y_slider.setObjectName("y_slider")
        self.z_slider = QtWidgets.QSlider(self.centralwidget)
        self.z_slider.setGeometry(QtCore.QRect(40, 240, 211, 31))
        self.z_slider.setMaximum(1000)
        self.z_slider.setProperty("value", 300)
        self.z_slider.setOrientation(QtCore.Qt.Horizontal)
        self.z_slider.setObjectName("z_slider")
        self.c_slider = QtWidgets.QSlider(self.centralwidget)
        self.c_slider.setGeometry(QtCore.QRect(40, 440, 211, 31))
        self.c_slider.setMaximum(360)
        self.c_slider.setOrientation(QtCore.Qt.Horizontal)
        self.c_slider.setObjectName("c_slider")
        self.a_slider = QtWidgets.QSlider(self.centralwidget)
        self.a_slider.setGeometry(QtCore.QRect(40, 310, 211, 31))
        self.a_slider.setMaximum(360)
        self.a_slider.setProperty("value", 180)
        self.a_slider.setOrientation(QtCore.Qt.Horizontal)
        self.a_slider.setObjectName("a_slider")
        self.b_slider = QtWidgets.QSlider(self.centralwidget)
        self.b_slider.setGeometry(QtCore.QRect(40, 380, 211, 31))
        self.b_slider.setMaximum(360)
        self.b_slider.setOrientation(QtCore.Qt.Horizontal)
        self.b_slider.setObjectName("b_slider")
        self.labelItemsList = QtWidgets.QListView(self.centralwidget)
        self.labelItemsList.setGeometry(QtCore.QRect(1610, 0, 291, 341))
        self.labelItemsList.setObjectName("labelItemsList")
        self.creat_new = QtWidgets.QPushButton(self.centralwidget)
        self.creat_new.setGeometry(QtCore.QRect(80, 540, 141, 51))
        self.creat_new.setObjectName("creat_new")
        self.saveButton = QtWidgets.QPushButton(self.centralwidget)
        self.saveButton.setGeometry(QtCore.QRect(80, 620, 141, 51))
        self.saveButton.setObjectName("saveButton")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(40, 690, 1881, 331))
        self.groupBox.setObjectName("groupBox")
        self.MPV_BD037_BUICK_GL8_2016 = QtWidgets.QRadioButton(self.groupBox)
        self.MPV_BD037_BUICK_GL8_2016.setGeometry(QtCore.QRect(0, 100, 231, 111))
        self.MPV_BD037_BUICK_GL8_2016.setText("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("model_images/MPV_BD037_BUICK_GL8_2016.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.MPV_BD037_BUICK_GL8_2016.setIcon(icon)
        self.MPV_BD037_BUICK_GL8_2016.setIconSize(QtCore.QSize(160, 160))
        self.MPV_BD037_BUICK_GL8_2016.setObjectName("MPV_BD037_BUICK_GL8_2016")
        self.SUV_BD269_Volkswagen_Tiguan_2017 = QtWidgets.QRadioButton(self.groupBox)
        self.SUV_BD269_Volkswagen_Tiguan_2017.setGeometry(QtCore.QRect(0, 210, 231, 111))
        self.SUV_BD269_Volkswagen_Tiguan_2017.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("model_images/SUV_BD207_Benz_GLC_2017.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.SUV_BD269_Volkswagen_Tiguan_2017.setIcon(icon1)
        self.SUV_BD269_Volkswagen_Tiguan_2017.setIconSize(QtCore.QSize(160, 160))
        self.SUV_BD269_Volkswagen_Tiguan_2017.setObjectName("SUV_BD269_Volkswagen_Tiguan_2017")
        self.Coupe_BD331_BMW_Z4 = QtWidgets.QRadioButton(self.groupBox)
        self.Coupe_BD331_BMW_Z4.setGeometry(QtCore.QRect(180, 0, 231, 111))
        self.Coupe_BD331_BMW_Z4.setText("")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("model_images/Coupe_BD331_BMW_Z4.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.Coupe_BD331_BMW_Z4.setIcon(icon2)
        self.Coupe_BD331_BMW_Z4.setIconSize(QtCore.QSize(160, 160))
        self.Coupe_BD331_BMW_Z4.setObjectName("Coupe_BD331_BMW_Z4")
        self.MPV_BD063_JIANGling_quanshun = QtWidgets.QRadioButton(self.groupBox)
        self.MPV_BD063_JIANGling_quanshun.setGeometry(QtCore.QRect(180, 100, 231, 111))
        self.MPV_BD063_JIANGling_quanshun.setText("")
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap("model_images/MPV_BD063_JIANGling_quanshun.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.MPV_BD063_JIANGling_quanshun.setIcon(icon3)
        self.MPV_BD063_JIANGling_quanshun.setIconSize(QtCore.QSize(160, 160))
        self.MPV_BD063_JIANGling_quanshun.setObjectName("MPV_BD063_JIANGling_quanshun")
        self.SUV_BD372_Nissan_Paladin = QtWidgets.QRadioButton(self.groupBox)
        self.SUV_BD372_Nissan_Paladin.setGeometry(QtCore.QRect(180, 210, 231, 111))
        self.SUV_BD372_Nissan_Paladin.setText("")
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap("model_images/SUV_BD372_Nissan_Paladin.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.SUV_BD372_Nissan_Paladin.setIcon(icon4)
        self.SUV_BD372_Nissan_Paladin.setIconSize(QtCore.QSize(160, 160))
        self.SUV_BD372_Nissan_Paladin.setObjectName("SUV_BD372_Nissan_Paladin")
        self.Coupe_BD347_Smart_ForTwo_2014 = QtWidgets.QRadioButton(self.groupBox)
        self.Coupe_BD347_Smart_ForTwo_2014.setGeometry(QtCore.QRect(360, 0, 231, 111))
        self.Coupe_BD347_Smart_ForTwo_2014.setText("")
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap("model_images/Coupe_BD347_Smart_ForTwo_2014.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.Coupe_BD347_Smart_ForTwo_2014.setIcon(icon5)
        self.Coupe_BD347_Smart_ForTwo_2014.setIconSize(QtCore.QSize(160, 160))
        self.Coupe_BD347_Smart_ForTwo_2014.setObjectName("Coupe_BD347_Smart_ForTwo_2014")
        self.MPV_BD374_Volkswagen_Transporter = QtWidgets.QRadioButton(self.groupBox)
        self.MPV_BD374_Volkswagen_Transporter.setGeometry(QtCore.QRect(360, 100, 231, 111))
        self.MPV_BD374_Volkswagen_Transporter.setText("")
        icon6 = QtGui.QIcon()
        icon6.addPixmap(QtGui.QPixmap("model_images/MPV_BD374_Volkswagen_Transporter.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.MPV_BD374_Volkswagen_Transporter.setIcon(icon6)
        self.MPV_BD374_Volkswagen_Transporter.setIconSize(QtCore.QSize(160, 160))
        self.MPV_BD374_Volkswagen_Transporter.setObjectName("MPV_BD374_Volkswagen_Transporter")
        self.SUV_BD449_Ford_Explorer = QtWidgets.QRadioButton(self.groupBox)
        self.SUV_BD449_Ford_Explorer.setGeometry(QtCore.QRect(360, 210, 231, 111))
        self.SUV_BD449_Ford_Explorer.setText("")
        icon7 = QtGui.QIcon()
        icon7.addPixmap(QtGui.QPixmap("model_images/SUV_BD449_Ford_Explorer.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.SUV_BD449_Ford_Explorer.setIcon(icon7)
        self.SUV_BD449_Ford_Explorer.setIconSize(QtCore.QSize(160, 160))
        self.SUV_BD449_Ford_Explorer.setObjectName("SUV_BD449_Ford_Explorer")
        self.Coupe_BD429_Porsche_Panamera_4S_2014 = QtWidgets.QRadioButton(self.groupBox)
        self.Coupe_BD429_Porsche_Panamera_4S_2014.setGeometry(QtCore.QRect(540, 0, 231, 111))
        self.Coupe_BD429_Porsche_Panamera_4S_2014.setText("")
        icon8 = QtGui.QIcon()
        icon8.addPixmap(QtGui.QPixmap("model_images/Coupe_BD429_Porsche_Panamera_4S_2014.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.Coupe_BD429_Porsche_Panamera_4S_2014.setIcon(icon8)
        self.Coupe_BD429_Porsche_Panamera_4S_2014.setIconSize(QtCore.QSize(160, 160))
        self.Coupe_BD429_Porsche_Panamera_4S_2014.setObjectName("Coupe_BD429_Porsche_Panamera_4S_2014")
        self.Notchback_BD216_Peugeot207 = QtWidgets.QRadioButton(self.groupBox)
        self.Notchback_BD216_Peugeot207.setGeometry(QtCore.QRect(540, 100, 231, 111))
        self.Notchback_BD216_Peugeot207.setText("")
        icon9 = QtGui.QIcon()
        icon9.addPixmap(QtGui.QPixmap("model_images/Notchback_BD216_Peugeot207.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.Notchback_BD216_Peugeot207.setIcon(icon9)
        self.Notchback_BD216_Peugeot207.setIconSize(QtCore.QSize(160, 160))
        self.Notchback_BD216_Peugeot207.setObjectName("Notchback_BD216_Peugeot207")
        self.Hatchback_BD002_SUZUKI_SWIFT_2016 = QtWidgets.QRadioButton(self.groupBox)
        self.Hatchback_BD002_SUZUKI_SWIFT_2016.setGeometry(QtCore.QRect(720, 0, 231, 111))
        self.Hatchback_BD002_SUZUKI_SWIFT_2016.setText("")
        icon10 = QtGui.QIcon()
        icon10.addPixmap(QtGui.QPixmap("model_images/Hatchback_BD002_SUZUKI_SWIFT_2016.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.Hatchback_BD002_SUZUKI_SWIFT_2016.setIcon(icon10)
        self.Hatchback_BD002_SUZUKI_SWIFT_2016.setIconSize(QtCore.QSize(160, 160))
        self.Hatchback_BD002_SUZUKI_SWIFT_2016.setObjectName("Hatchback_BD002_SUZUKI_SWIFT_2016")
        self.Notchback_BD341_Lexus_ES_2013 = QtWidgets.QRadioButton(self.groupBox)
        self.Notchback_BD341_Lexus_ES_2013.setGeometry(QtCore.QRect(720, 100, 231, 111))
        self.Notchback_BD341_Lexus_ES_2013.setText("")
        icon11 = QtGui.QIcon()
        icon11.addPixmap(QtGui.QPixmap("model_images/Notchback_BD341_Lexus_ES_2013.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.Notchback_BD341_Lexus_ES_2013.setIcon(icon11)
        self.Notchback_BD341_Lexus_ES_2013.setIconSize(QtCore.QSize(160, 160))
        self.Notchback_BD341_Lexus_ES_2013.setObjectName("Notchback_BD341_Lexus_ES_2013")
        self.Hatchback_BD306_BMW_1_series = QtWidgets.QRadioButton(self.groupBox)
        self.Hatchback_BD306_BMW_1_series.setGeometry(QtCore.QRect(900, 0, 231, 111))
        self.Hatchback_BD306_BMW_1_series.setText("")
        icon12 = QtGui.QIcon()
        icon12.addPixmap(QtGui.QPixmap("model_images/Hatchback_BD306_BMW_1_series.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.Hatchback_BD306_BMW_1_series.setIcon(icon12)
        self.Hatchback_BD306_BMW_1_series.setIconSize(QtCore.QSize(160, 160))
        self.Hatchback_BD306_BMW_1_series.setObjectName("Hatchback_BD306_BMW_1_series")
        self.Notchback_BD409_Volvo_S60_2013 = QtWidgets.QRadioButton(self.groupBox)
        self.Notchback_BD409_Volvo_S60_2013.setGeometry(QtCore.QRect(900, 100, 231, 111))
        self.Notchback_BD409_Volvo_S60_2013.setText("")
        icon13 = QtGui.QIcon()
        icon13.addPixmap(QtGui.QPixmap("model_images/Notchback_BD409_Volvo_S60_2013.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.Notchback_BD409_Volvo_S60_2013.setIcon(icon13)
        self.Notchback_BD409_Volvo_S60_2013.setIconSize(QtCore.QSize(160, 160))
        self.Notchback_BD409_Volvo_S60_2013.setObjectName("Notchback_BD409_Volvo_S60_2013")
        self.Hatchback_BD365_Ford_Fiesta_ST = QtWidgets.QRadioButton(self.groupBox)
        self.Hatchback_BD365_Ford_Fiesta_ST.setGeometry(QtCore.QRect(1080, 0, 231, 111))
        self.Hatchback_BD365_Ford_Fiesta_ST.setText("")
        icon14 = QtGui.QIcon()
        icon14.addPixmap(QtGui.QPixmap("model_images/Hatchback_BD365_Ford_Fiesta_ST.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.Hatchback_BD365_Ford_Fiesta_ST.setIcon(icon14)
        self.Hatchback_BD365_Ford_Fiesta_ST.setIconSize(QtCore.QSize(160, 160))
        self.Hatchback_BD365_Ford_Fiesta_ST.setObjectName("Hatchback_BD365_Ford_Fiesta_ST")
        self.SUV_BD036_BaoJUN730 = QtWidgets.QRadioButton(self.groupBox)
        self.SUV_BD036_BaoJUN730.setGeometry(QtCore.QRect(540, 210, 231, 111))
        self.SUV_BD036_BaoJUN730.setText("")
        icon15 = QtGui.QIcon()
        icon15.addPixmap(QtGui.QPixmap("model_images/SUV_BD036_BaoJUN730.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.SUV_BD036_BaoJUN730.setIcon(icon15)
        self.SUV_BD036_BaoJUN730.setIconSize(QtCore.QSize(160, 160))
        self.SUV_BD036_BaoJUN730.setObjectName("SUV_BD036_BaoJUN730")
        self.Hatchback_BD381_Fiat_Bravo_2011 = QtWidgets.QRadioButton(self.groupBox)
        self.Hatchback_BD381_Fiat_Bravo_2011.setGeometry(QtCore.QRect(1260, 0, 231, 111))
        self.Hatchback_BD381_Fiat_Bravo_2011.setText("")
        icon16 = QtGui.QIcon()
        icon16.addPixmap(QtGui.QPixmap("model_images/Hatchback_BD381_Fiat_Bravo_2011.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.Hatchback_BD381_Fiat_Bravo_2011.setIcon(icon16)
        self.Hatchback_BD381_Fiat_Bravo_2011.setIconSize(QtCore.QSize(160, 160))
        self.Hatchback_BD381_Fiat_Bravo_2011.setObjectName("Hatchback_BD381_Fiat_Bravo_2011")
        self.SUV_BD207_Benz_GLC_2017 = QtWidgets.QRadioButton(self.groupBox)
        self.SUV_BD207_Benz_GLC_2017.setGeometry(QtCore.QRect(720, 210, 231, 111))
        self.SUV_BD207_Benz_GLC_2017.setText("")
        self.SUV_BD207_Benz_GLC_2017.setIcon(icon1)
        self.SUV_BD207_Benz_GLC_2017.setIconSize(QtCore.QSize(160, 160))
        self.SUV_BD207_Benz_GLC_2017.setObjectName("SUV_BD207_Benz_GLC_2017")
        self.radioButton_9 = QtWidgets.QRadioButton(self.groupBox)
        self.radioButton_9.setGeometry(QtCore.QRect(540, 320, 231, 111))
        self.radioButton_9.setText("")
        icon17 = QtGui.QIcon()
        icon17.addPixmap(QtGui.QPixmap("../../.designer/backup/model_images/SUV_BD449_Ford_Explorer.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.radioButton_9.setIcon(icon17)
        self.radioButton_9.setIconSize(QtCore.QSize(160, 160))
        self.radioButton_9.setObjectName("radioButton_9")
        self.label_8 = QtWidgets.QLabel(self.groupBox)
        self.label_8.setGeometry(QtCore.QRect(200, 80, 171, 16))
        self.label_8.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_8.setAlignment(QtCore.Qt.AlignCenter)
        self.label_8.setObjectName("label_8")
        self.label_9 = QtWidgets.QLabel(self.groupBox)
        self.label_9.setGeometry(QtCore.QRect(370, 80, 171, 16))
        self.label_9.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_9.setAlignment(QtCore.Qt.AlignCenter)
        self.label_9.setObjectName("label_9")
        self.label_10 = QtWidgets.QLabel(self.groupBox)
        self.label_10.setGeometry(QtCore.QRect(560, 80, 171, 16))
        self.label_10.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_10.setAlignment(QtCore.Qt.AlignCenter)
        self.label_10.setObjectName("label_10")
        self.label_11 = QtWidgets.QLabel(self.groupBox)
        self.label_11.setGeometry(QtCore.QRect(740, 80, 171, 16))
        self.label_11.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_11.setAlignment(QtCore.Qt.AlignCenter)
        self.label_11.setObjectName("label_11")
        self.label_12 = QtWidgets.QLabel(self.groupBox)
        self.label_12.setGeometry(QtCore.QRect(910, 80, 171, 16))
        self.label_12.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_12.setAlignment(QtCore.Qt.AlignCenter)
        self.label_12.setObjectName("label_12")
        self.label_13 = QtWidgets.QLabel(self.groupBox)
        self.label_13.setGeometry(QtCore.QRect(560, 300, 171, 16))
        self.label_13.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_13.setAlignment(QtCore.Qt.AlignCenter)
        self.label_13.setObjectName("label_13")
        self.label_14 = QtWidgets.QLabel(self.groupBox)
        self.label_14.setGeometry(QtCore.QRect(20, 190, 171, 16))
        self.label_14.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_14.setAlignment(QtCore.Qt.AlignCenter)
        self.label_14.setObjectName("label_14")
        self.label_15 = QtWidgets.QLabel(self.groupBox)
        self.label_15.setGeometry(QtCore.QRect(190, 190, 171, 16))
        self.label_15.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_15.setAlignment(QtCore.Qt.AlignCenter)
        self.label_15.setObjectName("label_15")
        self.label_16 = QtWidgets.QLabel(self.groupBox)
        self.label_16.setGeometry(QtCore.QRect(380, 190, 171, 16))
        self.label_16.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_16.setAlignment(QtCore.Qt.AlignCenter)
        self.label_16.setObjectName("label_16")
        self.label_17 = QtWidgets.QLabel(self.groupBox)
        self.label_17.setGeometry(QtCore.QRect(560, 190, 171, 16))
        self.label_17.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_17.setAlignment(QtCore.Qt.AlignCenter)
        self.label_17.setObjectName("label_17")
        self.label_18 = QtWidgets.QLabel(self.groupBox)
        self.label_18.setGeometry(QtCore.QRect(740, 190, 171, 16))
        self.label_18.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_18.setAlignment(QtCore.Qt.AlignCenter)
        self.label_18.setObjectName("label_18")
        self.label_19 = QtWidgets.QLabel(self.groupBox)
        self.label_19.setGeometry(QtCore.QRect(910, 190, 171, 16))
        self.label_19.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_19.setAlignment(QtCore.Qt.AlignCenter)
        self.label_19.setObjectName("label_19")
        self.label_20 = QtWidgets.QLabel(self.groupBox)
        self.label_20.setGeometry(QtCore.QRect(20, 290, 171, 16))
        self.label_20.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_20.setAlignment(QtCore.Qt.AlignCenter)
        self.label_20.setObjectName("label_20")
        self.label_21 = QtWidgets.QLabel(self.groupBox)
        self.label_21.setGeometry(QtCore.QRect(190, 290, 171, 16))
        self.label_21.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_21.setAlignment(QtCore.Qt.AlignCenter)
        self.label_21.setObjectName("label_21")
        self.label_22 = QtWidgets.QLabel(self.groupBox)
        self.label_22.setGeometry(QtCore.QRect(380, 290, 171, 16))
        self.label_22.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_22.setAlignment(QtCore.Qt.AlignCenter)
        self.label_22.setObjectName("label_22")
        self.label_24 = QtWidgets.QLabel(self.groupBox)
        self.label_24.setGeometry(QtCore.QRect(1100, 80, 171, 16))
        self.label_24.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_24.setAlignment(QtCore.Qt.AlignCenter)
        self.label_24.setObjectName("label_24")
        self.label_25 = QtWidgets.QLabel(self.groupBox)
        self.label_25.setGeometry(QtCore.QRect(1280, 80, 171, 16))
        self.label_25.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_25.setAlignment(QtCore.Qt.AlignCenter)
        self.label_25.setObjectName("label_25")
        self.label_26 = QtWidgets.QLabel(self.groupBox)
        self.label_26.setGeometry(QtCore.QRect(740, 300, 171, 16))
        self.label_26.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_26.setAlignment(QtCore.Qt.AlignCenter)
        self.label_26.setObjectName("label_26")
        self.Coupe_BD001__Zotye_E200 = QtWidgets.QRadioButton(self.groupBox)
        self.Coupe_BD001__Zotye_E200.setGeometry(QtCore.QRect(0, 0, 231, 111))
        self.Coupe_BD001__Zotye_E200.setText("")
        icon18 = QtGui.QIcon()
        icon18.addPixmap(QtGui.QPixmap("model_images/Coupe_BD001__Zotye_E200.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.Coupe_BD001__Zotye_E200.setIcon(icon18)
        self.Coupe_BD001__Zotye_E200.setIconSize(QtCore.QSize(160, 160))
        self.Coupe_BD001__Zotye_E200.setObjectName("Coupe_BD001__Zotye_E200")
        self.label_7 = QtWidgets.QLabel(self.groupBox)
        self.label_7.setGeometry(QtCore.QRect(10, 80, 171, 16))
        self.label_7.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label_7.setObjectName("label_7")
        self.Hatchback_07 = QtWidgets.QRadioButton(self.groupBox)
        self.Hatchback_07.setGeometry(QtCore.QRect(1440, 100, 231, 111))
        self.Hatchback_07.setText("")
        icon19 = QtGui.QIcon()
        icon19.addPixmap(QtGui.QPixmap("model_images/Hatchback_07.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.Hatchback_07.setIcon(icon19)
        self.Hatchback_07.setIconSize(QtCore.QSize(160, 160))
        self.Hatchback_07.setObjectName("Hatchback_07")
        self.Hatchback_09 = QtWidgets.QRadioButton(self.groupBox)
        self.Hatchback_09.setGeometry(QtCore.QRect(1620, 210, 231, 111))
        self.Hatchback_09.setText("")
        icon20 = QtGui.QIcon()
        icon20.addPixmap(QtGui.QPixmap("model_images/Hatchback_09.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.Hatchback_09.setIcon(icon20)
        self.Hatchback_09.setIconSize(QtCore.QSize(160, 160))
        self.Hatchback_09.setObjectName("Hatchback_09")
        self.label_30 = QtWidgets.QLabel(self.groupBox)
        self.label_30.setGeometry(QtCore.QRect(1640, 290, 171, 16))
        self.label_30.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_30.setAlignment(QtCore.Qt.AlignCenter)
        self.label_30.setObjectName("label_30")
        self.Notchback_01 = QtWidgets.QRadioButton(self.groupBox)
        self.Notchback_01.setGeometry(QtCore.QRect(1080, 100, 231, 111))
        self.Notchback_01.setText("")
        icon21 = QtGui.QIcon()
        icon21.addPixmap(QtGui.QPixmap("model_images/Notchback_01.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.Notchback_01.setIcon(icon21)
        self.Notchback_01.setIconSize(QtCore.QSize(160, 160))
        self.Notchback_01.setObjectName("Notchback_01")
        self.label_31 = QtWidgets.QLabel(self.groupBox)
        self.label_31.setGeometry(QtCore.QRect(1130, 880, 171, 16))
        self.label_31.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_31.setAlignment(QtCore.Qt.AlignCenter)
        self.label_31.setObjectName("label_31")
        self.label_29 = QtWidgets.QLabel(self.groupBox)
        self.label_29.setGeometry(QtCore.QRect(1100, 180, 171, 16))
        self.label_29.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_29.setAlignment(QtCore.Qt.AlignCenter)
        self.label_29.setObjectName("label_29")
        self.Notchback_12 = QtWidgets.QRadioButton(self.groupBox)
        self.Notchback_12.setGeometry(QtCore.QRect(1260, 100, 231, 111))
        self.Notchback_12.setText("")
        icon22 = QtGui.QIcon()
        icon22.addPixmap(QtGui.QPixmap("model_images/Notchback_12.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.Notchback_12.setIcon(icon22)
        self.Notchback_12.setIconSize(QtCore.QSize(160, 160))
        self.Notchback_12.setObjectName("Notchback_12")
        self.label_32 = QtWidgets.QLabel(self.groupBox)
        self.label_32.setGeometry(QtCore.QRect(1280, 180, 171, 16))
        self.label_32.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_32.setAlignment(QtCore.Qt.AlignCenter)
        self.label_32.setObjectName("label_32")
        self.label_33 = QtWidgets.QLabel(self.groupBox)
        self.label_33.setGeometry(QtCore.QRect(1470, 180, 171, 16))
        self.label_33.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_33.setAlignment(QtCore.Qt.AlignCenter)
        self.label_33.setObjectName("label_33")
        self.Hatchback_05 = QtWidgets.QRadioButton(self.groupBox)
        self.Hatchback_05.setGeometry(QtCore.QRect(1440, 210, 231, 111))
        self.Hatchback_05.setText("")
        icon23 = QtGui.QIcon()
        icon23.addPixmap(QtGui.QPixmap("model_images/Hatchback_05.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.Hatchback_05.setIcon(icon23)
        self.Hatchback_05.setIconSize(QtCore.QSize(160, 160))
        self.Hatchback_05.setObjectName("Hatchback_05")
        self.SUV_06 = QtWidgets.QRadioButton(self.groupBox)
        self.SUV_06.setGeometry(QtCore.QRect(900, 210, 231, 111))
        self.SUV_06.setText("")
        icon24 = QtGui.QIcon()
        icon24.addPixmap(QtGui.QPixmap("model_images/SUV_06.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.SUV_06.setIcon(icon24)
        self.SUV_06.setIconSize(QtCore.QSize(160, 160))
        self.SUV_06.setObjectName("SUV_06")
        self.label_34 = QtWidgets.QLabel(self.groupBox)
        self.label_34.setGeometry(QtCore.QRect(980, 290, 54, 12))
        self.label_34.setObjectName("label_34")
        self.Hatchback_04 = QtWidgets.QRadioButton(self.groupBox)
        self.Hatchback_04.setGeometry(QtCore.QRect(1260, 210, 231, 111))
        self.Hatchback_04.setText("")
        icon25 = QtGui.QIcon()
        icon25.addPixmap(QtGui.QPixmap("model_images/Hatchback_04.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.Hatchback_04.setIcon(icon25)
        self.Hatchback_04.setIconSize(QtCore.QSize(160, 160))
        self.Hatchback_04.setObjectName("Hatchback_04")
        self.label_27 = QtWidgets.QLabel(self.groupBox)
        self.label_27.setGeometry(QtCore.QRect(1260, 290, 171, 16))
        self.label_27.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_27.setAlignment(QtCore.Qt.AlignCenter)
        self.label_27.setObjectName("label_27")
        self.label_28 = QtWidgets.QLabel(self.groupBox)
        self.label_28.setGeometry(QtCore.QRect(1440, 290, 171, 16))
        self.label_28.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_28.setAlignment(QtCore.Qt.AlignCenter)
        self.label_28.setObjectName("label_28")
        self.Hatchback_BD073_Skoda_Fabia = QtWidgets.QRadioButton(self.groupBox)
        self.Hatchback_BD073_Skoda_Fabia.setGeometry(QtCore.QRect(1440, 0, 231, 111))
        self.Hatchback_BD073_Skoda_Fabia.setText("")
        icon26 = QtGui.QIcon()
        icon26.addPixmap(QtGui.QPixmap("model_images/Hatchback_BD073_Skoda_Fabia.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.Hatchback_BD073_Skoda_Fabia.setIcon(icon26)
        self.Hatchback_BD073_Skoda_Fabia.setIconSize(QtCore.QSize(160, 160))
        self.Hatchback_BD073_Skoda_Fabia.setObjectName("Hatchback_BD073_Skoda_Fabia")
        self.label_23 = QtWidgets.QLabel(self.groupBox)
        self.label_23.setGeometry(QtCore.QRect(1473, 80, 131, 20))
        self.label_23.setObjectName("label_23")
        self.MPV_16 = QtWidgets.QRadioButton(self.groupBox)
        self.MPV_16.setGeometry(QtCore.QRect(1080, 210, 231, 111))
        self.MPV_16.setText("")
        icon27 = QtGui.QIcon()
        icon27.addPixmap(QtGui.QPixmap("model_images/MPV_16.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.MPV_16.setIcon(icon27)
        self.MPV_16.setIconSize(QtCore.QSize(160, 160))
        self.MPV_16.setObjectName("MPV_16")
        self.label_63 = QtWidgets.QLabel(self.groupBox)
        self.label_63.setGeometry(QtCore.QRect(1100, 290, 171, 16))
        self.label_63.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_63.setAlignment(QtCore.Qt.AlignCenter)
        self.label_63.setObjectName("label_63")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(280, 120, 54, 12))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(280, 190, 54, 12))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(280, 250, 54, 12))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(280, 320, 54, 12))
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(280, 390, 54, 12))
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(280, 450, 54, 12))
        self.label_6.setObjectName("label_6")
        self.graphicsView = GraphicsView(self.centralwidget)
        self.graphicsView.setGeometry(QtCore.QRect(330, 20, 1261, 651))
        self.graphicsView.setObjectName("graphicsView")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1920, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.creat_new.setText(_translate("MainWindow", "NewCar"))
        self.saveButton.setText(_translate("MainWindow", "Save"))
        self.groupBox.setTitle(_translate("MainWindow", "GroupBox"))
        self.label_8.setText(_translate("MainWindow", "Coupe_BD331_BMW_Z4"))
        self.label_9.setText(_translate("MainWindow", "Coupe_BD347_Smart_ForTwo"))
        self.label_10.setText(_translate("MainWindow", "Coupe_BD429_Porsche_Panamera"))
        self.label_11.setText(_translate("MainWindow", "Hatchback_SUZUKI_SWIFT"))
        self.label_12.setText(_translate("MainWindow", "Hatchback_BMW_1_series"))
        self.label_13.setText(_translate("MainWindow", "SUV__BaoJUN730"))
        self.label_14.setText(_translate("MainWindow", "MPVBUICK_GL8_2016"))
        self.label_15.setText(_translate("MainWindow", "MPV_JIANGling_quanshun"))
        self.label_16.setText(_translate("MainWindow", "MPV_Volkswagen_Transporter"))
        self.label_17.setText(_translate("MainWindow", "Notchback_Peugeot207"))
        self.label_18.setText(_translate("MainWindow", "Notchback_Lexus_ES_2013"))
        self.label_19.setText(_translate("MainWindow", "Notchback_Volvo_S60_2013"))
        self.label_20.setText(_translate("MainWindow", "SUV_Volkswagen_Tiguan_2017"))
        self.label_21.setText(_translate("MainWindow", "SUV_Nissan_Paladin"))
        self.label_22.setText(_translate("MainWindow", "SUV_BD449_Ford_Explorer"))
        self.label_24.setText(_translate("MainWindow", "Hatchback_Ford_Fiesta_ST"))
        self.label_25.setText(_translate("MainWindow", "Hatchback_Fiat_Bravo_2011"))
        self.label_26.setText(_translate("MainWindow", "SUV_Benz_GLC_2017"))
        self.label_7.setText(_translate("MainWindow", "Coupe_BD001__Zotye_E200"))
        self.label_30.setText(_translate("MainWindow", "Hatchback_09"))
        self.label_31.setText(_translate("MainWindow", "Hatchback_07"))
        self.label_29.setText(_translate("MainWindow", "Notchback_01"))
        self.label_32.setText(_translate("MainWindow", "Notchback_12"))
        self.label_33.setText(_translate("MainWindow", "Hatchback_07"))
        self.label_34.setText(_translate("MainWindow", "SUV_06"))
        self.label_27.setText(_translate("MainWindow", "Hatchback_04"))
        self.label_28.setText(_translate("MainWindow", "Hatchback_05"))
        self.label_23.setText(_translate("MainWindow", "Hatchback_Skoda_Fabia"))
        self.label_63.setText(_translate("MainWindow", "MPV_16"))
        self.label.setText(_translate("MainWindow", "X"))
        self.label_2.setText(_translate("MainWindow", "Y"))
        self.label_3.setText(_translate("MainWindow", "Z"))
        self.label_4.setText(_translate("MainWindow", "a"))
        self.label_5.setText(_translate("MainWindow", "b"))
        self.label_6.setText(_translate("MainWindow", "c"))