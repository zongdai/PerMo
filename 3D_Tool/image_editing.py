import cv2
import numpy as np
import math
import time
import random
def write_pointcloud(write_path, vertexs,rgb = None):
    with open(write_path, 'w') as f:
        for i in range(0, len(vertexs)//3):
            if rgb is None:
                f.write("v " + str(vertexs[i*3]) + " " + str(vertexs[i*3+1]) + " " + str(vertexs[i*3+2]) + "\n")
            else:
                f.write("v " + str(vertexs[i*3]) + " " + str(vertexs[i*3+1]) + " " + str(vertexs[i*3+2]) + " " + str(rgb[i*3]) + " " + str(rgb[i*3+1]) + " " + str(rgb[i*3+2]) + "\n")


def write_pointcloud_np(write_path, vertexs_np, rgbs = None):
    # print(vertexs_np.shape)
    # print(len(rgb))
    with open(write_path, 'w') as f:
        for k, i in enumerate(range(0, vertexs_np.shape[1])):
            if rgbs is None:
                f.write("v " + str(vertexs_np[0][i]) + " " + str(vertexs_np[1][i]) + " " + str(vertexs_np[2][i]) + "\n")
            else:
                # print(rgb[k])
                # f.write("v " + str(vertexs_np[0][i]) + " " + str(vertexs_np[1][i]) + " " + str(vertexs_np[2][i]) + " " + str(rgb[k][0]) + " " + str(rgb[k][1]) + " " + str([k][2]) + "\n")
                f.write("v " + str(vertexs_np[0][i]) + " " + str(vertexs_np[1][i]) + " " + str(vertexs_np[2][i]) + " " + str(rgbs[0][i]) + " " + str(rgbs[1][i]) + " " + str(rgbs[2][i]) + "\n")




def read_pointcloud(point_cloud_path):
    x = []
    y = []
    z = []
    try:
        with open(point_cloud_path) as f:
            line = f.readline()

            while (line):
                if random.randint(0,100) < 90:
                    line = f.readline()
                    continue
                if line[0] == 'v' and line[1] == ' ':
                    items = line.split(' ')
                    x.append(float(items[1]))
                    y.append(float(items[2]))
                    z.append(float(items[3].split('\n')[0]))
                line = f.readline()
    except FileNotFoundError:
        x.append(-100.0)
        y.append(-100.0)
        z.append(-100.0)
    finally:
        return x ,y ,z



def read_pointcloud_with_t(point_cloud_path, tx, ty, tz):

    with open(point_cloud_path) as f:
        line = f.readline()
        vertexs = []
        rgbs = []
        while (line):
            vertex = []
            rgb = []
            if line[0] == 'v':
                items = line.split(' ')
                vertex.append(float(items[1]) - tx)
                vertex.append(float(items[2]) - ty)
                vertex.append(float(items[3]) - tz)
                rgb.append(float(items[4]) * 255)
                rgb.append(float(items[5]) * 255)
                rgb.append(float(items[6].split('\n')[0]) * 255)
                # vertex.append(1.0)
                vertexs.append(vertex)
                rgbs.append(rgb)
            line = f.readline()
    return np.array(vertexs).T, rgbs

def read_mesh(point_cloud_path):
    
    x = []
    y = []
    z = []
    faces = []
    try:
        with open(point_cloud_path) as f:
            line = f.readline()
            items = line.split(' ')
            while (line):
                if line[0] == 'v' and line[1] == ' ':
                    x.append(float(items[1]))
                    y.append(float(items[2]))
                    z.append(float(items[3].split('\n')[0]))
                   
                if line[0] == 'f':
                    face = []
                    face.append(float(items[1].split("/")[0]))
                    face.append(float(items[2].split("/")[0]))
                    face.append(float(items[3].split('/')[0]))
                    if len(items) == 6:
                        face.append(float(items[4].split('/')[0]))
                    faces.append(face)
                line = f.readline()
                items = line.split(' ')
    except FileNotFoundError:
        print("Can not read model")
        print(point_cloud_path)
    finally:
        return np.array([x, y, z]), faces

def get_rotation_mat(a, b, c):
    rotation = np.zeros((3,3))
    rotation[0][0] = math.cos(c) * math.cos(b)
    rotation[0][1] = -math.sin(c) * math.cos(a) + math.cos(c) * math.sin(b) * math.sin(a)
    rotation[0][2] = math.sin(a) * math.sin(c) + math.cos(c) * math.sin(b) * math.cos(a)
    rotation[1][0] = math.cos(b) * math.sin(c)
    rotation[1][1] = math.cos(c) * math.cos(a) + math.sin(c) * math.sin(b) * math.sin(a)
    rotation[1][2] = -math.sin(a) * math.cos(c) + math.cos(a) * math.sin(b) * math.sin(c)
    rotation[2][0] = -math.sin(b)
    rotation[2][1] = math.cos(b) * math.sin(a)
    rotation[2][2] = math.cos(a) * math.cos(b)
    return rotation
# pose(0, 0) = cos(c) * cos(b); pose(0, 1) = -sin(c) * cos(a) + cos(c) * sin(b) * sin(a); pose(0, 2) = sin(a) * sin(c) + cos(c) * sin(b) * cos(a);
# 			pose(1, 0) = cos(b) * sin(c);	 pose(1, 1) = cos(c) * cos(a) + sin(c) * sin(b) * sin(a);   pose(1, 2) = -sin(a) * cos(c) + cos(a) * sin(b) * sin(c);
# 			pose(2, 0) = -sin(b);			 pose(2, 1) = cos(b) * sin(a);			                            pose(2, 2) = cos(a) * cos(b);
# 			pose(3, 0) = 0;			         pose(3, 1) = 0;			                                            pose(3, 2) = 0;
def fill_ori_part(img, point_3d, projection_matrix, rgbs = None):
    p = np.dot(projection_matrix, point_3d)
    u = p[0, :] / p[2, :]
    v = p[1, :] / p[2, :]
    for i, j in zip(u, v):
        # print(i,j)
        # if abs(i) >= 1920 or abs(j) >= 1080:
        #     continue
        cv2.circle(img, (int(i), int(j)), 1, (0, 0, 0, 0.2), -1)

def projection2img(img, maskimg, car_index , name_index, part_index, new_point_3d, ori_point_3d, projection_matrix, rgbs, rotation_axis_global, orientation, angel):
    if(part_index != 1):
        return
    fill_ori_part(img, ori_point_3d, projection_matrix) # need modify
    theat = 10
    if part_index == 2:
        view_orientation = np.array([rotation_axis_global['x'], rotation_axis_global['y'], rotation_axis_global['z']]).T
        # print(orientation)
        # print(view_orientation)
        theat = math.acos(np.dot(orientation.T, view_orientation) / np.sqrt(np.dot(orientation.T, orientation) * np.dot(view_orientation.T, view_orientation)))
        print("theat = " + str(theat/math.pi * 180))


    deform_part_img, min_u, min_v = project_and_process_deform_part(new_point_3d, projection_matrix, rgbs, [img.shape[0], img.shape[1]])
    if deform_part_img is None:
        return img

    for i in range(deform_part_img.shape[0]):
        for j in range(deform_part_img.shape[1]):
            if deform_part_img[i][j][3] == 0:
                continue

            u = j + min_u
            v = i + min_v
            if angel < theat:
                img[v][u][0] = deform_part_img[i][j][0]
                img[v][u][1] = deform_part_img[i][j][1]
                img[v][u][2] = deform_part_img[i][j][2]
            else:
                img[v][u][0] = 40
                img[v][u][1] = 40
                img[v][u][2] = 40

            maskimg[v][u][0] = car_index
    class_mask = (maskimg[:, :, 0] == car_index)
    # print(class_mask.shape)
    maskimg[class_mask, 1] = 255 - part_index
    maskimg[class_mask, 2] = name_index
    #
    # return img

# def project_and_process_unvisible_door()

def project_and_process_deform_part(point_3d, projection_matrix, rgbs, ori_img_shape):
    print(ori_img_shape)
    img_bg = np.zeros((ori_img_shape[0], ori_img_shape[1], 4))
    p = np.dot(projection_matrix, point_3d)
    # write_pointcloud_np("taxi\\fine\\123.obj", point_3d)
    u = p[0, :] / p[2, :]
    v = p[1, :] / p[2, :]
    max_u = min(int(np.max(u)), ori_img_shape[1])
    min_u = max(int(np.min(u)), 0)
    max_v = min(int(np.max(v)), ori_img_shape[0])
    min_v = max(int(np.min(v)), 0)
    if min_u > max_u or min_v > max_v:
        return None, min_u, min_v
    k = 0
    for i, j in zip(u, v):
        # BGR
        # print(rgbs[k][1])
        if j >= img_bg.shape[0] or i >= img_bg.shape[1]:
            continue
        # print(i,j)
        img_bg[int(j)][int(i)][0] = int(rgbs[2][k])
        img_bg[int(j)][int(i)][1] = int(rgbs[1][k])
        img_bg[int(j)][int(i)][2] = int(rgbs[0][k])
        img_bg[int(j)][int(i)][3] = 255
        k+=1
    # show_img = cv2.resize(img_bg, (800,500))
    cv2.imwrite("back\\bg.png", img_bg)
    # cv2.waitKey()

    border = []
    # find border
    for i in range(min_v, max_v+1):
        start_flag = False
        start = 0
        end = 0

        for j in range(min_u, max_u+1):
            # find start
            if i >= img_bg.shape[0] - 1 or j >= img_bg.shape[1] - 1:
                continue
            while(start_flag == False):
                # print(img_bg[i][j][3])
                if img_bg[i][j][3] == 255:
                    start = j
                    start_flag = True
                    break
                j+=1
                if j >= img_bg.shape[1] - 1:
                    break

            # find end
            if start_flag:
                if img_bg[i][j][3] == 255:
                    end = j

        border.append([start, end])

    # fill the holes
    for k, i in enumerate(range(min_v, max_v)):
        for j in range(border[k][0], border[k][1]+1):
            if img_bg[i][j][3] == 0:
                    vaild_pix_count = 0.0
                    if i - 1 > 0 and j - 1 > 0 and                                       img_bg[i - 1][j - 1][3] == 255:
                        vaild_pix_count += 1
                        img_bg[i][j][0] += img_bg[i-1][j-1][0]
                        img_bg[i][j][1] += img_bg[i-1][j-1][1]
                        img_bg[i][j][2] += img_bg[i-1][j-1][2]
                    if i + 1 < ori_img_shape[0] and j - 1 > 0 and                    img_bg[i + 1][j - 1][3] == 255:
                        vaild_pix_count += 1
                        img_bg[i][j][0] += img_bg[i + 1][j - 1][0]
                        img_bg[i][j][1] += img_bg[i + 1][j - 1][1]
                        img_bg[i][j][2] += img_bg[i + 1][j - 1][2]
                    if i - 1 > 0 and j + 1 < ori_img_shape[1] - 1 and                    img_bg[i - 1][j + 1][3] == 255:
                        vaild_pix_count += 1
                        img_bg[i][j][0] += img_bg[i - 1][j + 1][0]
                        img_bg[i][j][1] += img_bg[i - 1][j + 1][1]
                        img_bg[i][j][2] += img_bg[i - 1][j + 1][2]
                    if i + 1 < ori_img_shape[0] and j + 1 < ori_img_shape[1] and img_bg[i + 1][j + 1][3] == 255:
                        vaild_pix_count += 1
                        img_bg[i][j][0] += img_bg[i + 1][j + 1][0]
                        img_bg[i][j][1] += img_bg[i + 1][j + 1][1]
                        img_bg[i][j][2] += img_bg[i + 1][j + 1][2]
                    if j - 1 > 0 and                                                     img_bg[i][j - 1][3] == 255:
                        vaild_pix_count += 1
                        img_bg[i][j][0] += img_bg[i][j - 1][0]
                        img_bg[i][j][1] += img_bg[i][j - 1][1]
                        img_bg[i][j][2] += img_bg[i][j - 1][2]
                    if j + 1 < ori_img_shape[1] and                                  img_bg[i][j + 1][3] == 255:
                        vaild_pix_count += 1
                        img_bg[i][j][0] += img_bg[i][j + 1][0]
                        img_bg[i][j][1] += img_bg[i][j + 1][1]
                        img_bg[i][j][2] += img_bg[i][j + 1][2]
                    if i - 1 > 0 and                                                     img_bg[i - 1][j][3] == 255:
                        vaild_pix_count += 1
                        img_bg[i][j][0] += img_bg[i-1][j][0]
                        img_bg[i][j][1] += img_bg[i-1][j][1]
                        img_bg[i][j][2] += img_bg[i-1][j][2]
                    if i + 1 < ori_img_shape[0] and                                                     img_bg[i + 1][j][3] == 255:
                        vaild_pix_count += 1
                        img_bg[i][j][0] += img_bg[i + 1][j][0]
                        img_bg[i][j][1] += img_bg[i + 1][j][1]
                        img_bg[i][j][2] += img_bg[i + 1][j][2]
                    if vaild_pix_count > 0:
                        img_bg[i][j][0] = img_bg[i][j][0] / vaild_pix_count
                        img_bg[i][j][1] = img_bg[i][j][1] / vaild_pix_count
                        img_bg[i][j][2] = img_bg[i][j][2] / vaild_pix_count
                        img_bg[i][j][3] = 255

    result = img_bg[min_v:max_v+1, min_u:max_u+1, :]
    if result.shape[0] <= 0 or result.shape[1] <= 0:
        return None, min_u, min_v
    result = cv2.dilate(result, (5, 5), iterations=1)
    result = cv2.erode(result, (5, 5), iterations=1)
    cv2.imwrite("back\\result.png", result)
    return result, min_u, min_v

def project_and_process_deform_part_for_pix3d(u, v, rgbs, ori_img_shape):
    print(ori_img_shape)
    img_bg = np.zeros((ori_img_shape[0], ori_img_shape[1], 4))
    # write_pointcloud_np("taxi\\fine\\123.obj", point_3d)

    max_u = min(int(np.max(u)), ori_img_shape[1])
    min_u = max(int(np.min(u)), 0)
    max_v = min(int(np.max(v)), ori_img_shape[0])
    min_v = max(int(np.min(v)), 0)
    if min_u > max_u or min_v > max_v:
        return None, min_u, min_v
    k = 0
    for i, j in zip(u, v):
        # BGR
        # print(rgbs[k][1])
        if j >= img_bg.shape[0] or i >= img_bg.shape[1]:
            continue
        # print(i,j)
        img_bg[int(j)][int(i)][0] = int(rgbs[2][k])
        img_bg[int(j)][int(i)][1] = int(rgbs[1][k])
        img_bg[int(j)][int(i)][2] = int(rgbs[0][k])
        img_bg[int(j)][int(i)][3] = 255
        k+=1

    border = []
    # find border
    for i in range(min_v, max_v+1):
        start_flag = False
        start = 0
        end = 0

        for j in range(min_u, max_u+1):
            # find start
            if i >= img_bg.shape[0] - 1 or j >= img_bg.shape[1] - 1:
                continue
            while(start_flag == False):
                # print(img_bg[i][j][3])
                if img_bg[i][j][3] == 255:
                    start = j
                    start_flag = True
                    break
                j+=1
                if j >= img_bg.shape[1] - 1:
                    break

            # find end
            if start_flag:
                if img_bg[i][j][3] == 255:
                    end = j

        border.append([start, end])

    # fill the holes
    for k, i in enumerate(range(min_v, max_v)):
        for j in range(border[k][0], border[k][1]+1):
            if img_bg[i][j][3] == 0:
                    vaild_pix_count = 0.0
                    if i - 1 > 0 and j - 1 > 0 and                                       img_bg[i - 1][j - 1][3] == 255:
                        vaild_pix_count += 1
                        img_bg[i][j][0] += img_bg[i-1][j-1][0]
                        img_bg[i][j][1] += img_bg[i-1][j-1][1]
                        img_bg[i][j][2] += img_bg[i-1][j-1][2]
                    if i + 1 < ori_img_shape[0] and j - 1 > 0 and                    img_bg[i + 1][j - 1][3] == 255:
                        vaild_pix_count += 1
                        img_bg[i][j][0] += img_bg[i + 1][j - 1][0]
                        img_bg[i][j][1] += img_bg[i + 1][j - 1][1]
                        img_bg[i][j][2] += img_bg[i + 1][j - 1][2]
                    if i - 1 > 0 and j + 1 < ori_img_shape[1] - 1 and                    img_bg[i - 1][j + 1][3] == 255:
                        vaild_pix_count += 1
                        img_bg[i][j][0] += img_bg[i - 1][j + 1][0]
                        img_bg[i][j][1] += img_bg[i - 1][j + 1][1]
                        img_bg[i][j][2] += img_bg[i - 1][j + 1][2]
                    if i + 1 < ori_img_shape[0] and j + 1 < ori_img_shape[1] and img_bg[i + 1][j + 1][3] == 255:
                        vaild_pix_count += 1
                        img_bg[i][j][0] += img_bg[i + 1][j + 1][0]
                        img_bg[i][j][1] += img_bg[i + 1][j + 1][1]
                        img_bg[i][j][2] += img_bg[i + 1][j + 1][2]
                    if j - 1 > 0 and                                                     img_bg[i][j - 1][3] == 255:
                        vaild_pix_count += 1
                        img_bg[i][j][0] += img_bg[i][j - 1][0]
                        img_bg[i][j][1] += img_bg[i][j - 1][1]
                        img_bg[i][j][2] += img_bg[i][j - 1][2]
                    if j + 1 < ori_img_shape[1] and                                  img_bg[i][j + 1][3] == 255:
                        vaild_pix_count += 1
                        img_bg[i][j][0] += img_bg[i][j + 1][0]
                        img_bg[i][j][1] += img_bg[i][j + 1][1]
                        img_bg[i][j][2] += img_bg[i][j + 1][2]
                    if i - 1 > 0 and                                                     img_bg[i - 1][j][3] == 255:
                        vaild_pix_count += 1
                        img_bg[i][j][0] += img_bg[i-1][j][0]
                        img_bg[i][j][1] += img_bg[i-1][j][1]
                        img_bg[i][j][2] += img_bg[i-1][j][2]
                    if i + 1 < ori_img_shape[0] and                                                     img_bg[i + 1][j][3] == 255:
                        vaild_pix_count += 1
                        img_bg[i][j][0] += img_bg[i + 1][j][0]
                        img_bg[i][j][1] += img_bg[i + 1][j][1]
                        img_bg[i][j][2] += img_bg[i + 1][j][2]
                    if vaild_pix_count > 0:
                        img_bg[i][j][0] = img_bg[i][j][0] / vaild_pix_count
                        img_bg[i][j][1] = img_bg[i][j][1] / vaild_pix_count
                        img_bg[i][j][2] = img_bg[i][j][2] / vaild_pix_count
                        img_bg[i][j][3] = 255

    result = img_bg[min_v:max_v+1, min_u:max_u+1, :]
    if result.shape[0] <= 0 or result.shape[1] <= 0:
        return None, min_u, min_v
    result = cv2.dilate(result, (5, 5), iterations=1)
    result = cv2.erode(result, (5, 5), iterations=1)
    cv2.imwrite("indoor\\blend.png", result)
    return result, min_u, min_v

def rotate_part(rot_mat, part, t_x, t_y, t_z):
    part_rot = np.dot(rot_mat, part)
    part_ori = np.copy(part)
    part_ori[0, :] = part[0, :] + t_x
    part_ori[1, :] = part[1, :] + t_y
    part_ori[2, :] = part[2, :] + t_z
    part_rot[0, :] = part_rot[0, :] + t_x
    part_rot[1, :] = part_rot[1, :] + t_y
    part_rot[2, :] = part_rot[2, :] + t_z

    return part_rot, part_ori

def get_car_name(index):
    model_name = {}
    model_name[0] = "baojun-310-2017"
    model_name[1] = "biaozhi-3008"
    model_name[2] = "biaozhi-liangxiang"
    model_name[3] = "bieke-yinglang-XT"
    model_name[4] = "biyadi-2x-F0"
    model_name[5] = "changanbenben"
    model_name[6] = "dongfeng-DS5"
    model_name[7] = "feiyate"
    model_name[8] = "fengtian-liangxiang"
    model_name[9] = "fengtian-MPV"
    model_name[10] = "jilixiongmao-2015"
    model_name[11] = "lingmu-aotuo-2009"
    model_name[12] = "lingmu-swift"
    model_name[13] = "lingmu-SX4-2012"
    model_name[14] = "sikeda-jingrui"
    model_name[15] = "fengtian-weichi-2006"
    model_name[16] = "037-CAR02"
    model_name[17] = "aodi-a6"
    model_name[18] = "baoma-330"
    model_name[19] = "baoma-530"
    model_name[20] = "baoshijie-paoche"
    model_name[21] = "bentian-fengfan"
    model_name[22] = "biaozhi-408"
    model_name[23] = "biaozhi-508"
    model_name[24] = "bieke-kaiyue"
    model_name[25] = "fute"
    model_name[26] = "haima-3"
    model_name[27] = "kaidilake-CTS"
    model_name[28] = "leikesasi"
    model_name[29] = "mazida-6-2015"
    model_name[30] = "MG-GT-2015"
    model_name[31] = "oubao"
    model_name[32] = "qiya"
    model_name[33] = "rongwei-750"
    model_name[34] = "supai-2016"
    model_name[35] = "xiandai-suonata"
    model_name[36] = "yiqi-benteng-b50"
    model_name[37] = "bieke"
    model_name[38] = "biyadi-F3"
    model_name[39] = "biyadi-qin"
    model_name[40] = "dazhong"
    model_name[41] = "dazhongmaiteng"
    model_name[42] = "dihao-EV"
    model_name[43] = "dongfeng-xuetielong-C6"
    model_name[44] = "dongnan-V3-lingyue-2011"
    model_name[45] = "dongfeng-yulong-naruijie"
    model_name[46] = "019-SUV"
    model_name[47] = "036-CAR01"
    model_name[48] = "aodi-Q7-SUV"
    model_name[49] = "baojun-510"
    model_name[50] = "baoma-X5"
    model_name[51] = "baoshijie-kayan"
    model_name[52] = "beiqi-huansu-H3"
    model_name[53] = "benchi-GLK-300"
    model_name[54] = "benchi-ML500"
    model_name[55] = "fengtian-puladuo-06"
    model_name[56] = "fengtian-SUV-gai"
    model_name[57] = "guangqi-chuanqi-GS4-2015"
    model_name[58] = "jianghuai-ruifeng-S3"
    model_name[59] = "jili-boyue"
    model_name[60] = "jipu-3"
    model_name[61] = "linken-SUV"
    model_name[62] = "lufeng-X8"
    model_name[63] = "qirui-ruihu"
    model_name[64] = "rongwei-RX5"
    model_name[65] = "sanling-oulande"
    model_name[66] = "sikeda-SUV"
    model_name[67] = "Skoda_Fabia-2011"
    model_name[68] = "xiandai-i25-2016"
    model_name[69] = "yingfeinidi-qx80"
    model_name[70] = "yingfeinidi-SUV"
    model_name[71] = "benchi-SUR"
    model_name[72] = "biyadi-tang"
    model_name[73] = "changan-CS35-2012"
    model_name[74] = "changan-cs5"
    model_name[75] = "changcheng-H6-2016"
    model_name[76] = "dazhong-SUV"
    model_name[77] = "dongfeng-fengguang-S560"
    model_name[78] = "dongfeng-fengxing-SX6"
    return model_name[index]



def read_pose_file(pose_file):

    with open(pose_file) as f:
        car_infos = []
        line = f.readline()
        while(line):
            info = {}
            items = line.split(' ')
            # print(items)
            info["name_index"] = int(items[0])
            info["a"] = float(items[1])
            info["b"] = float(items[2])
            info["c"] = float(items[3])
            info["tx"] = float(items[4])
            info["ty"] = float(items[5])
            info["tz"] = float(items[6].split('\n')[0])
            # print(info)
            car_infos.append(info)
            line = f.readline()
    return car_infos
# folder = "front\\"
# img_name = "171206_035346603_Camera_5"
#
# depth_img = cv2.imread(folder + img_name + "-depth.png",cv2.IMREAD_ANYDEPTH)
# depth_map = depth_img / 100.0
# row, col = depth_map.shape
# color_img = cv2.imread(folder + img_name + ".jpg")
#
# label_img = cv2.imread(folder + img_name + "-label.png")
# point_3d = []
# rgb = []
# fx = 2304.54786556982
# fy = 2305.875668062
# cx = 1686.23787612802
# cy = 1354.98486439791
#
#
#
# for i in range(row):
#     for j in range(col):
#         if depth_map[i][j] == 500:
#             continue
#         if label_img[i][j][0] != 1:
#             continue
#
#         x = (j - cx) * depth_map[i][j] / fx
#         y = (i + 1 - row + cy) * depth_map[i][j] / fy
#         z = depth_map[i][j]
#
#         point_3d.append(x)
#         point_3d.append(y)
#         point_3d.append(z)
#
#         # b g r sequence
#         rgb.append(color_img[i][j][2])
#         rgb.append(color_img[i][j][1])
#         rgb.append(color_img[i][j][0])
# #
# # write_pointcloud(folder + "pointcloud2.obj", point_3d, rgb)
#
# x, y, z = read_pointcloud(folder + "rotation.obj")
# t_x = np.mean(np.array(x))
# t_y = np.mean(np.array(y))
# t_z = np.mean(np.array(z))
#
#
# # back_door set t_x = 0
# # four door set t_y = 0
# t_y = 0
# door, rgbs = read_pointcloud_with_t(folder + "door2.obj", t_x, t_y, t_z)
#
# camera_mat = np.array([[fx, 0, cx],
#                        [0, fy, cy],
#                        [0, 0 , 1 ]])
#
# for i in range(1, 30):
#     time_start = time.time()
#
#     rot_mat = get_rotation_mat(-2*i*math.pi/180, 0, 0)
#     door_rot, door_ori = rotate_part(rot_mat, door, t_x, t_y, t_z)
#     # write_pointcloud_np("door_rot.obj", door_rot, rgb=rgbs)
#     result_img = projection2img(folder + img_name + ".jpg", door_rot, door_ori, camera_mat, rgbs)
#     cv2.imwrite(folder + "result\\" + img_name + "_" + str(2*i) + ".png", result_img)
#     time_end = time.time()
#     print('cost', time_end - time_start)