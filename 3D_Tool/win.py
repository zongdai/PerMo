import sys
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QDockWidget, QListWidget, QGraphicsPixmapItem, QGraphicsScene,  QMessageBox
from PyQt5.QtGui import *
import os
from image_editing import read_pointcloud, get_rotation_mat, get_car_name
from vis import read_model, get_part_patch_box, convert_texture_map_to_uv_map
from test import Ui_MainWindow  # 导入创建的GUI类
import numpy as np
import math
# pyinstaller -F win.py --noconsole
class mywindow(QtWidgets.QMainWindow, Ui_MainWindow):

    def __init__(self):
        super(mywindow, self).__init__()
        self.setupUi(self)
        self.creat_new.clicked.connect(self.creat_new_car)
        self.saveButton.clicked.connect(self.save_annotation)
        self.x_slider.valueChanged.connect(self.valuechange)
        self.y_slider.valueChanged.connect(self.valuechange)
        self.z_slider.valueChanged.connect(self.valuechange)
        self.a_slider.valueChanged.connect(self.valuechange)
        self.b_slider.valueChanged.connect(self.valuechange)
        self.c_slider.valueChanged.connect(self.valuechange)

        # 绑定模型选择控件
        self.Coupe_BD001__Zotye_E200.toggled.connect(lambda: self.select_car_type(self.Coupe_BD001__Zotye_E200))
        self.Coupe_BD331_BMW_Z4.toggled.connect(lambda: self.select_car_type(self.Coupe_BD331_BMW_Z4))
        self.Coupe_BD347_Smart_ForTwo_2014.toggled.connect(lambda: self.select_car_type(self.Coupe_BD347_Smart_ForTwo_2014))
        self.Coupe_BD429_Porsche_Panamera_4S_2014.toggled.connect(lambda: self.select_car_type(self.Coupe_BD429_Porsche_Panamera_4S_2014))
        self.Hatchback_BD002_SUZUKI_SWIFT_2016.toggled.connect(lambda: self.select_car_type(self.Hatchback_BD002_SUZUKI_SWIFT_2016))
        self.Hatchback_BD073_Skoda_Fabia.toggled.connect(lambda: self.select_car_type(self.Hatchback_BD073_Skoda_Fabia))
        self.Hatchback_BD306_BMW_1_series.toggled.connect(lambda: self.select_car_type(self.Hatchback_BD306_BMW_1_series))
        self.Hatchback_BD365_Ford_Fiesta_ST.toggled.connect(lambda: self.select_car_type(self.Hatchback_BD365_Ford_Fiesta_ST))
        self.Hatchback_BD381_Fiat_Bravo_2011.toggled.connect(lambda: self.select_car_type(self.Hatchback_BD381_Fiat_Bravo_2011))
        self.MPV_BD037_BUICK_GL8_2016.toggled.connect(lambda: self.select_car_type(self.MPV_BD037_BUICK_GL8_2016))
        self.MPV_BD063_JIANGling_quanshun.toggled.connect(lambda: self.select_car_type(self.MPV_BD063_JIANGling_quanshun))
        self.MPV_BD374_Volkswagen_Transporter.toggled.connect(lambda: self.select_car_type(self.MPV_BD374_Volkswagen_Transporter))
        self.Notchback_BD216_Peugeot207.toggled.connect(lambda: self.select_car_type(self.Notchback_BD216_Peugeot207))
        self.Notchback_BD341_Lexus_ES_2013.toggled.connect(lambda: self.select_car_type(self.Notchback_BD341_Lexus_ES_2013))
        self.Notchback_BD409_Volvo_S60_2013.toggled.connect(lambda: self.select_car_type(self.Notchback_BD409_Volvo_S60_2013))
        self.SUV_BD036_BaoJUN730.toggled.connect(lambda: self.select_car_type(self.SUV_BD036_BaoJUN730))
        self.SUV_BD207_Benz_GLC_2017.toggled.connect(lambda: self.select_car_type(self.SUV_BD207_Benz_GLC_2017))
        self.SUV_BD269_Volkswagen_Tiguan_2017.toggled.connect(lambda: self.select_car_type(self.SUV_BD269_Volkswagen_Tiguan_2017))
        self.SUV_BD372_Nissan_Paladin.toggled.connect(lambda: self.select_car_type(self.SUV_BD372_Nissan_Paladin))
        self.SUV_BD449_Ford_Explorer.toggled.connect(lambda: self.select_car_type(self.SUV_BD449_Ford_Explorer))

        self.SUV_06.toggled.connect(lambda: self.select_car_type(self.SUV_06))
        self.Hatchback_04.toggled.connect(lambda: self.select_car_type(self.Hatchback_04))
        self.Hatchback_05.toggled.connect(lambda: self.select_car_type(self.Hatchback_05))
        self.Hatchback_07.toggled.connect(lambda: self.select_car_type(self.Hatchback_07))
        self.Hatchback_09.toggled.connect(lambda: self.select_car_type(self.Hatchback_09))
        self.Notchback_01.toggled.connect(lambda: self.select_car_type(self.Notchback_01))
        self.Notchback_12.toggled.connect(lambda: self.select_car_type(self.Notchback_12))
        self.MPV_16.toggled.connect(lambda: self.select_car_type(self.MPV_16))

        # 标注结果存储
        self.annotations = []
        # 标注的车辆存储
        self.car_pcs = []
        # 所有车辆的点云模型
        self.models = {}
        # 当前选择的标注编号
        self.select_an_index = 0
        # 当前选择的车辆类型
        self.current_car_tpye = 0
        self.camera_mat = None
        self.car_names = []
        self.meshes_pc = {}
        self.us = {}
        self.vs = {}
        self.u_maps = {}
        self.v_maps = {}
        self.faces = {}

        # 初始化
        self.load_all_pc()

        # 初始化
        self.init_fileList()

        self.name2index = {}
        names = os.listdir('models')
        names.sort()
        for index, name in enumerate(names):
            self.name2index[name.split('.')[0]] = index
        print(self.name2index)


    def init_fileList(self):
        # 读取所有待标注的图像
        files = os.listdir('images')
        listModel = QStringListModel()
        listModel.setStringList(files)

        self.filesList.setModel(listModel)
        # self.image_files = files
        # 图像文件列表绑定点击动作
        self.filesList.clicked.connect(self.check_file_item)
        # 将第一个文件标记为当前标注文件
        self.current_img_name = files[0].split('.')[0]
        self.files = files

    def init_labelList(self):
        '''
        初始化已标注的文件
        在点击图像列表时触发
        '''
        items = self.get_labelitems()

        listModel = QStringListModel()
        listModel.setStringList(items)

        self.labelItemsList.setModel(listModel)
        self.labelItemsList.clicked.connect(self.click_label_item)


    def reload_image(self):
        # image2show = np.copy(self.image)
        # image_height, image_width, image_depth = image2show.shape
        # print(image2show.shape)
        # QIm = cv2.cvtColor(image2show, cv2.COLOR_BGR2RGB)
        # QIm = QImage(QIm.data, image_width, image_height,
        #              image_width * image_depth,
        #              QImage.Format_RGB888)
        # self.imageView.setPixmap(QPixmap.fromImage(QIm))
        # cv2.imshow('win', image2show)

        # cv2.imshow('win', self.image)
        QIm = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        # cv2.imshow('win', self.image)
        x_scroball, y_scroball = self.graphicsView.get_scroball()
        x_scroball_max, y_scroball_max = self.graphicsView.get_max_scroball()
        self.graphicsView.load_img(QIm, x_scroball, y_scroball, x_scroball_max, y_scroball_max)

    def add_labelList(self):
        '''
        当创建一个新的标注车辆时，在label list 新建一项
        :return:
        '''
        items = [str(i+1) for i in range(len(self.annotations))]
        listModel = QStringListModel()
        listModel.setStringList(items)
        self.labelItemsList.setModel(listModel)

    def get_labelitems(self):
        '''
        获取已标注的内容
        :return:
        '''
        items = []
        if os.path.exists('label_result/' + self.current_img_name + '.txt'):
            with open('label_result/' + self.current_img_name + '.txt') as f:
                line = f.readline()
                index = 1
                while line:
                    [car_type, x, y, z, a, b, c] = [float(item) for item in line.split()]
                    if b < 0:
                        b += math.pi * 2
                    self.current_car_tpye = int(car_type)
                    self.annotations.append([a, b, c, x, y, z, car_type])
                    items.append(str(index))
                    index += 1
                    line = f.readline()
        return items


    def valuechange(self):
        '''
        当任意滑动条发生变化时，读取所有数值，重新加载图像
        '''
        self.x = (self.x_slider.value()/100)
        self.y = (self.y_slider.value()/(-100))
        self.z = self.z_slider.value() / 10
        self.a = self.a_slider.value() / 180.0 * math.pi
        self.b = self.b_slider.value() / 180.0 * math.pi
        self.c = self.c_slider.value() / 180.0 * math.pi
        self.tanslate_draw(self.a, self.b, self.c, self.x, self.y, self.z, self.current_car_tpye, [0, 255, 0])
        self.annotations[self.select_an_index] = [self.a, self.b, self.c, self.x, self.y, self.z, self.current_car_tpye]

        self.reload_image()


    def check_file_item(self, index):
        '''
        点击图像列表时，读取相应图像并显示
        '''

        # self.save_annotation()

        # 读取相机参数

        image = cv2.imread('images/' + self.files[index.row()])
        # image = image.astype(np.uint8)
        self.get_camera_mat(self.files[index.row()])

        self.ori_image = np.copy(image)
        self.image_bg = np.copy(image)
        self.image = image
        # self.ori_image = cv2.resize(image, (1024, 512))
        # self.image_bg = cv2.resize(image, (1024, 512))
        # self.image = cv2.resize(image, (1024, 512))
        self.current_img_name = self.files[index.row()].split('.')[0]
        self.select_an_index = 0
        self.annotations = []
        self.init_labelList()
        # self.draw_others(current_index=-1)
        self.reload_image()


    def click_label_item(self, index):
        '''
        点击标注好的每一项，
        重新初始化图像，显示已标注好的车辆
        改变滑动条的数值
        '''
        # print(len(self.annotations))
        self.image_bg = np.copy(self.ori_image)
        self.image = np.copy(self.ori_image)
        self.select_an_index = index.row()
        self.draw_others(current_index=self.select_an_index)
        print(self.annotations)
        a = self.annotations[self.select_an_index][0]
        b = self.annotations[self.select_an_index][1]
        c = self.annotations[self.select_an_index][2]
        x = self.annotations[self.select_an_index][3]
        y = self.annotations[self.select_an_index][4]
        z = self.annotations[self.select_an_index][5]
        # car_type = self.annotations[self.select_an_index][6]
        print(int(self.annotations[self.select_an_index][6]))
        self.current_car_tpye = int(self.annotations[self.select_an_index][6])
        self.x_slider.setValue(int(x*100))
        self.y_slider.setValue(int(y * (-100)))
        self.z_slider.setValue(int(z*10))
        self.a_slider.setValue(int(a * 180 / math.pi))
        self.b_slider.setValue(int(b * 180 / math.pi))
        self.c_slider.setValue(int(c * 180 / math.pi))

    def tanslate_draw(self, a, b, c, x, y, z, car_type, color, is_draw_bg = False):
        '''
        给定位姿，画出车辆
        '''
        camera_mat = self.camera_mat
        rot_mat = get_rotation_mat(a, b, c )
        pc = np.copy(self.models[self.car_names[int(car_type)]])


        pc = np.dot(rot_mat, pc)
        pc[0, :] += x
        pc[1, :] += y
        pc[2, :] += z

        p = np.dot(camera_mat, pc)
        u = p[0, :] / p[2, :]
        v = p[1, :] / p[2, :]
        self.image = np.copy(self.image_bg)
        for i, j in zip(u, v):
            if (i) >= self.image.shape[1] or (j) >= self.image.shape[0] or i < 0 or j < 0:
                continue
            if is_draw_bg:
                self.image_bg[int(j)][int(i)][0] = color[0]
                self.image_bg[int(j)][int(i)][1] = color[1]
                self.image_bg[int(j)][int(i)][2] = color[2]
            else:
                self.image[int(j)][int(i)][0] = color[0]
                self.image[int(j)][int(i)][1] = color[1]
                self.image[int(j)][int(i)][2] = color[2]
        colors = [(255, 0, 255), (0, 255, 255), (255, 200, 255)]
        if not is_draw_bg and z > 0:
            axis_img, mask = self.render(np.copy(self.meshes_pc[self.car_names[car_type]]), self.faces[self.car_names[car_type]],
                                         self.us[self.car_names[car_type]], self.vs[self.car_names[car_type]], x, y, z, a, b, c, camera_mat, self.image.shape[1],
                                         self.image.shape[0], colors)
            # print(self.image.shape)
            # print(axis_img.shape)
            self.image[:, :, 0][mask] = axis_img[:, :, 0][mask] // 2 + self.image[:, :, 0][mask] // 2
            self.image[:, :, 1][mask] = axis_img[:, :, 1][mask] // 2 + self.image[:, :, 1][mask] // 2
            self.image[:, :, 2][mask] = axis_img[:, :, 2][mask] // 2 + self.image[:, :, 2][mask] // 2


    def draw_others(self, current_index = -1):
        '''
        画出其他车辆，
        其他车辆指，非选定的车辆
        '''
        for i in range(len(self.annotations)):
            if i != current_index:
                an = self.annotations[i]
                # print(an)
                self.tanslate_draw(an[0], an[1], an[2], an[3], an[4], an[5], an[6], [255, 0, 0], is_draw_bg=True)

    def creat_new_car(self):
        '''
        创建一个新的标注车辆
        :return:
        '''
        self.car_pcs.append(self.models[self.car_names[0]])
        self.annotations.append([0, 0, 0, 0, 0, 0, 0])
        self.select_an_index = len(self.annotations) - 1
        self.image_bg = np.copy(self.ori_image)
        self.draw_others()
        self.reload_image()
        self.add_labelList()


    def save_annotation(self):

       if self.select_an_index >= 0:
           with open('label_result/' + self.current_img_name + '.txt', 'w')  as f:
               for an in self.annotations:
                   # print(an)
                   f.write(str(an[6]) + ' ' + str(an[3]) + ' ' + str(an[4]) + ' ' + str(an[5]) + ' ' + str(
                       an[0]) + ' ' + str(an[1]) + ' ' + str(an[2]) + '\n')
           QMessageBox.information(self, "QListView", "保存成功")

    def load_all_pc(self):
        obj_names = os.listdir('simplification_reorganize')
        obj_names.sort()
        print(obj_names)
        self.part_bboxs = get_part_patch_box('axis/Template18_new.PNG')
        rot = get_rotation_mat(math.pi/2, 0, 0)
        for obj_name in obj_names:
            name = obj_name.split('.')[0]
            # print(name)
            self.car_names.append(name)
            pc = read_pointcloud('models/' + obj_name)
            pc = np.dot(rot, pc)
            # pc[1,:] -= (np.max(pc[1,:]) + np.min(pc[1, :]))/2

            self.models[name] = pc
            pc, face_index, u, v, _ = read_model('simplification_reorganize/' + obj_name , read_face=True, read_vt=True)
            pc = np.dot(rot, pc)
            # pc[1,:] -= (np.max(pc[1,:]) + np.min(pc[1, :]))/2
            
            self.meshes_pc[name] = pc
            self.faces[name] = face_index
            self.us[name] = u
            self.vs[name] = v

            # u_map, v_map = convert_texture_map_to_uv_map(self.part_bboxs, 'axis/Template18_new.PNG')
            # self.u_maps[name] = u_map
            # self.v_maps[name] = v_map

    def select_car_type(self, btn):
        print(btn.objectName())
        # for k,v in self.name2index.items():
        #     print(k, v)
        # print(self.name2index[btn.objectName()])
        # print('...')
        if self.current_car_tpye != self.name2index[btn.objectName()]:
            self.current_car_tpye = self.name2index[btn.objectName()]

            self.annotations[self.select_an_index] = [self.a, self.b, self.c, self.x, self.y, self.z, self.name2index[btn.objectName()]]
            # self.annotations[self.select_an_index]
            self.tanslate_draw(self.a, self.b, self.c, self.x, self.y, self.z, self.current_car_tpye, [0, 255, 0])

            self.reload_image()

    def render(self, pc, face_index, t_u, t_v, x, y, z, a, b, c, camera_mat, width, height, colors):
        color_table = [
            [247, 77, 149],
            [32, 148, 9],
            [166, 104, 6],
            [7, 212, 133],
            [1, 251, 1],
            [2, 2, 188],
            [219, 251, 1],
            [96, 94, 92],
            [229, 114, 84],
            [216, 166, 255],
            [113, 165, 231],
            [8, 78, 183],
            [112, 252, 57],
            [5, 28, 126],
            [100, 111, 156],
            [140, 60, 39],
            [75, 13, 159],
            [188, 110, 83]
        ]
        tem_depth = np.ones((height, width, 3), dtype=np.uint16) * 20000
        depth_map = np.ones((height, width, 3), dtype=np.uint16) * 20000
        res = np.zeros((height, width, 3), dtype=np.uint8)
        tem = np.zeros((height, width, 3), dtype=np.uint8)
        if True:
            rot_mat = get_rotation_mat(a, b, c)
            pc2 = np.dot(rot_mat, pc)
            pc2[0, :] += x
            pc2[1, :] += y
            pc2[2, :] += z

            pc2 = np.dot(camera_mat, pc2)
            u = np.int32(pc2[0, :] / pc2[2, :]).reshape(1, -1)
            v = np.int32(pc2[1, :] / pc2[2, :]).reshape(1, -1)
            zz = pc2[2, :].reshape(1, -1)
            u_item = (u[0, face_index])  # shape is num_face * 3
            v_item = (v[0, face_index])  # shape is num_face * 3
            z_item = (zz[0, face_index])  # shape is num_face * 3

            max_us = np.max(u_item, axis=1)  # num_face * 1
            max_vs = np.max(v_item, axis=1)
            min_us = np.min(u_item, axis=1)
            min_vs = np.min(v_item, axis=1)
            face_depth = np.average(z_item, axis=1)

            t_u = t_u.reshape(1, -1)
            t_v = t_v.reshape(1, -1)
            t_u_item = (t_u[0, face_index])  # shape is num_face * 3
            t_v_item = (t_v[0, face_index])  # shape is num_face * 3

            max_tus = np.max(t_u_item, axis=1)
            max_tvs = np.max(t_v_item, axis=1)
            min_tus = np.min(t_u_item, axis=1)
            min_tvs = np.min(t_v_item, axis=1)

            for us, vs, min_v, max_v, min_u, max_u, des, min_tv, max_tv, min_tu, max_tu in zip(u_item, v_item, min_vs,
                                                                                               max_vs, min_us, max_us,
                                                                                               face_depth, min_tvs,
                                                                                               max_tvs,
                                                                                               min_tus, max_tus):
                part_index = 0
                for p, bbox in enumerate(self.part_bboxs):

                    if (min_tv + max_tv) / 2 > bbox[0] and (min_tu + max_tu) / 2 > bbox[1] and (min_tv + max_tv) / 2 < \
                            bbox[2] and (min_tu + max_tu) / 2 < bbox[3]:
                        part_index = p
                triangle = np.array([
                    [us[0], vs[0]],
                    [us[1], vs[1]],
                    [us[2], vs[2]]
                ], np.int32)

                des = int(des * 100)

                cv2.fillConvexPoly(tem_depth, triangle,
                                   (des, des, des))
                cv2.fillConvexPoly(tem, triangle,
                                   (color_table[part_index][2], color_table[part_index][1], color_table[part_index][0]))
                append_mask = tem_depth[min_v:max_v, min_u:max_u, 0:1] < depth_map[min_v:max_v, min_u:max_u, 0:1]
                depth_map[min_v:max_v, min_u:max_u, 0:1][append_mask] = des

                #
                res[min_v:max_v, min_u:max_u, 0:1][append_mask] = color_table[part_index][2]
                res[min_v:max_v, min_u:max_u, 1:2][append_mask] = color_table[part_index][1]
                res[min_v:max_v, min_u:max_u, 2:3][append_mask] = color_table[part_index][0]

        mask = ((res[:, :, 0] > 0) & (res[:, :, 1] > 0))

        return res, mask


    def get_camera_mat(self, file_name):
        # fx, fy, cx, cy = None
        fx = 721.53
        fy = 721.53
        cx = 609.559
        cy = 172.854
        print(file_name)
        with open('calib/' + file_name.split('.')[0] + '.txt') as f:
            line = f.readline()
            item = line.split()
            fx , cx, fy, cy = float(item[1]), float(item[3]), float(item[6]), float(item[7])
        camera_mat = np.array([[fx, 0, cx],
                               [0, fy, cy],
                               [0, 0, 1]])

        self.camera_mat = camera_mat

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = mywindow()
    window.show()
    sys.exit(app.exec_())
