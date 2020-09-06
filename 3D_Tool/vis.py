import cv2
import numpy as np
import time
import os
from multiprocessing import Process
import math
import json


def read_pose_file(pose_file):
    with open(pose_file) as f:
        car_infos = []
        line = f.readline()
        while (line):
            info = {}
            items = line.split(' ')
            # print(items)
            info["name"] = items[0]
            info["a"] = float(items[4]) + math.pi / 2
            info["b"] = float(items[5])
            info["c"] = float(items[6].split('\n')[0])
            info["tx"] = float(items[1])
            info["ty"] = float(items[2])
            info["tz"] = float(items[3])
            # print(info)
            car_infos.append(info)
            line = f.readline()
    return car_infos


def read_model(point_cloud_path, read_face=False, read_color=False, read_vt=False, scale=1.0):
    x = []
    y = []
    z = []
    u = []
    v = []
    rgb = []
    face_index = []
    with open(point_cloud_path) as f:
        line = f.readline()

        while line:
            if line[0] == 'v' and line[1] == ' ':
                items = line.split(' ')
                x.append(float(items[1]) * scale)
                y.append(float(items[2]) * scale)
                z.append(float(items[3].replace("\n", "")) * scale)
                if read_color:
                    r = float(items[4])
                    g = float(items[5])
                    b = float(items[6].replace("\n", ""))
                    rgb.append([r, g, b])

            elif line[0] == 'f' and read_face:
                item = line.split(' ')
                f1 = int(item[1].split('/')[0]) - 1
                f2 = int(item[2].split('/')[0]) - 1
                f3 = int(item[3].split('/')[0]) - 1
                if len(item) == 5:
                    f4 = int(item[4].split('/')[0]) - 1
                    face_index.append([f1, f2, f3, f4])
                else:
                    face_index.append([f1, f2, f3])
            elif line[0] == 'v' and line[1] == 't':
                items = line.split(' ')
                u.append(int((float(items[1])) * 2048))
                v.append(int((1 - float(items[2])) * 2048))
            line = f.readline()

    return np.array([x, y, z]), np.array(face_index), np.array(u), np.array(v), np.array(rgb).T


def get_texture_part_color():
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
        [113, 165, 0],
        [8, 78, 183],
        [112, 252, 57],
        [5, 28, 126],
        [100, 111, 156],
        [140, 60, 39],
        [75, 13, 159],
        [188, 110, 83]
    ]

    return color_table


def get_rotation_mat(a, b, c):
    matrix_a = np.array(
        [[1, 0, 0],
         [0, math.cos(a), -1 * math.sin(a)],
         [0, math.sin(a), math.cos(a)]]
    )
    matrix_b = np.array(
        [[math.cos(b), 0, math.sin(b)],
         [0, 1, 0],
         [-1 * math.sin(b), 0, math.cos(b)]
         ]
    )
    matrix_c = np.array(
        [[math.cos(c), -1 * math.sin(c), 0],
         [math.sin(c), math.cos(c), 0],
         [0, 0, 1]]
    )
    rotation_mat = np.dot(matrix_c, matrix_b)
    rotation_mat = np.dot(rotation_mat, matrix_a)
    return rotation_mat


def get_part_patch_box(texture_map_path):
    texture_map = cv2.imread(texture_map_path)
    parts_color = get_texture_part_color()
    bboxes = []
    for color in parts_color:
        area_uv = np.argwhere(texture_map[:, :, 2] == color[0])
        # print(area_uv.shape)
        min_v = np.min(area_uv[:, 0])
        min_u = np.min(area_uv[:, 1])
        max_v = np.max(area_uv[:, 0])
        max_u = np.max(area_uv[:, 1])
        bboxes.append([min_v, min_u, max_v, max_u])
    return bboxes


def convert_uv_to_part_uv(part_bboxes, us, vs):
    part_u = []
    part_v = []
    for u, v in zip(us, vs):
        for bbox in part_bboxes:
            min_v = bbox[0]
            min_u = bbox[1]
            max_v = bbox[2]
            max_u = bbox[3]

            if v > min_v and v < max_v and u > min_u and u < max_u:
                # print(u,v)
                part_u.append((u - min_u) / (max_u - min_u))
                part_v.append((v - min_v) / (max_v - min_v))

    return part_u, part_v


def convert_texture_map_to_uv_map(part_bboxes, texture_map_path):
    texture_map = cv2.imread(texture_map_path)
    u_map = np.zeros((texture_map.shape[0], texture_map.shape[1]))
    v_map = np.zeros((texture_map.shape[0], texture_map.shape[1]))
    for bbox in part_bboxes:
        min_v = bbox[0]
        min_u = bbox[1]
        max_v = bbox[2]
        max_u = bbox[3]
        for i in range(min_v, max_v):
            for j in range(min_u, max_u):
                v_map[i][j] = (i - min_v) / (max_v - min_v)
                u_map[i][j] = (j - min_u) / (max_u - min_u)
    return u_map, v_map


def append_uv_map(uv_map, img, alpha=0.5):
    heat_map = np.zeros((img.shape[0], img.shape[1], 3))
    #  color_bar = cv2.imread('bar.png')[3:, :, :]
    uv = np.argwhere(uv_map > 0)
    # print(uv.shape)
    for k in range(uv.shape[0]):
        i = uv[k][0]
        j = uv[k][1]
        heat_map[i][j][0] = int((1 - uv_map[i][j]) * 255)
        heat_map[i][j][1] = int((1 - uv_map[i][j]) * 255)
        heat_map[i][j][2] = int((1 - uv_map[i][j]) * 255)
    # heat_map[i][j][3] = 255

    append_area = uv_map > 0
    img[append_area] = heat_map[append_area] * alpha + img[append_area] * (1 - alpha)

    return img


def vis():
    uv_map = np.load('./labeled/uv_part_result/006039_u_map.npy') / 255
    img = cv2.imread('./labeled/result/006039.png')
    res = append_uv_map(uv_map, img, alpha=0.8)
    cv2.imwrite('vis.jpg', res)


def read_pcs(model_dir):
    names = [f for f in os.listdir(model_dir) if 'obj' in f]
    pcs = {}
    for name in names:
        pc, face_index, t_u, t_v, _ = read_model(os.path.join(model_dir, name), read_face=True, read_vt=True)
        pcs[name.split('.')[0]] = pc

    return pcs, face_index, t_u, t_v


def get_camera_mat(name):
    with open('./calib/' + name + '.txt') as f:
        line = f.readline()
        items = line.split(' ')
        fx = float(items[1])
        cx = float(items[3])
        fy = float(items[6])
        cy = float(items[7])

    return np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1.0]
    ])


def get_camera_matrix(pose_file):
    calib_file = os.path.join('./cityscape_camera_trainvaltest/camera/all/',
                              pose_file.split('_')[0] + '_' + pose_file.split('_')[1] + '_' + pose_file.split('_')[
                                  2] + '_camera' + '.json')
    with open(calib_file, 'r') as f:
        camera_info = json.load(f)
        fx = float(camera_info['intrinsic']['fx'])
        fy = float(camera_info['intrinsic']['fy'])
        cx = float(camera_info['intrinsic']['u0'])
        cy = float(camera_info['intrinsic']['v0'])
        camera_matrix = np.array(
            [[fx, 0, cx],
             [0, fy, cy],
             [0, 0, 1]]
        )
    return camera_matrix


def uv_ge(pose_files, uv_ouput_dir, pose_dir, model_dir, camera_mat, u_map, v_map, pcs, face_index, t_u, t_v,
          part_bboxs):
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
        [113, 165, 0],
        [8, 78, 183],
        [112, 252, 57],
        [5, 28, 126],
        [100, 111, 156],
        [140, 60, 39],
        [75, 13, 159],
        [188, 110, 83]
    ]
    for count, pose_file in enumerate(pose_files):
        print(str(count) + '/' + str(len(pose_files)))
        # oriimg = cv2.imread('/media/beta/Newsmy/Kitti_3D/kitti_3d/image/' + pose_file.split('.')[0] + '.png')
        # height, width, _ = oriimg.shape
        height = 375
        width = 1242
        depth_map = np.ones((height, width)) * 50000
        part_id = np.ones((height, width, 3), dtype=np.uint8) * 255
        u_map_res = np.zeros((height, width))
        v_map_res = np.zeros((height, width))
        # img = cv2.imread(os.path.join(raw_image_dir, '171206_034559609_Camera_5.jpg'))
        # camera_mat = get_camera_matrix(pose_file)
        pose_infos = read_pose_file(os.path.join(pose_dir, pose_file))
        for car_id, pose_info in enumerate(pose_infos):
            # model_name = get_car_name(pose_info["name_index"])
            # pc, face_index, t_u, t_v, _ = read_model(os.path.join(model_dir, model_name + '.obj'), read_face=True, read_vt=True)
            pc = pcs[pose_info['name']]
            rot_mat = get_rotation_mat(pose_info['a'], pose_info['b'], pose_info['c'])
            # rot_mat1 = get_rotation_mat(math.pi/2, 0, 0)
            # rot_mat2 = get_rotation_mat(math.pi, 0, 0)
            # pc2 = np.dot(rot_mat1, pc)
            # pc2 = np.dot(rot_mat2, pc)
            pc2 = np.dot(rot_mat, pc)
            # with open('/home/beta/cityscape_test/' + pose_file.split('.')[0] + str(car_id) + '.obj', 'w') as f:
            #     for i in range(pc2.shape[1]):
            #         f.write('v ' + str(pc2[0][i]) + ' ' + str(pc2[1][i]) + ' ' + str(pc2[2][i]) + '\n')
            pc2[0, :] += pose_info['tx']
            pc2[1, :] += pose_info['ty']
            pc2[2, :] += pose_info['tz']

            z = pc2[2, :].reshape(1, -1)

            pc2 = np.dot(camera_mat, pc2)

            u = np.int32(pc2[0, :] / pc2[2, :]).reshape(1, -1)
            v = np.int32(pc2[1, :] / pc2[2, :]).reshape(1, -1)
            # z = pc[2,:].reshape(1, -1)

            u_item = (u[0, face_index])  # shape is num_face * 3
            v_item = (v[0, face_index])  # shape is num_face * 3
            z_item = (z[0, face_index])  # shape is num_face * 3

            max_us = np.max(u_item, axis=1)
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

            for min_v, max_v, min_u, max_u, des, min_tv, max_tv, min_tu, max_tu in zip(min_vs, max_vs, min_us, max_us,
                                                                                       face_depth, min_tvs, max_tvs,
                                                                                       min_tus, max_tus):
                part_index = 0
                for p, bbox in enumerate(part_bboxs):
                    if min_tv > bbox[0] and min_tu > bbox[1] and max_tv < bbox[2] and max_tu < bbox[3]:
                        part_index = p
                if max_v >= height or max_u >= width or min_u < 0 or min_v < 0:
                    continue
                if max_v - min_v <= 0 or max_u - min_u <= 0 or max_tv - min_tv <= 0 or max_tu - min_tu <= 0 or (
                        max_tv - min_tv) * (max_tu - min_tu) > 100:
                    if max_v - min_v == 0 or max_u - min_u == 0:
                        if des < depth_map[max_v][max_u]:
                            u_map_res[max_v][max_u] = u_map[min_tv][min_tu]
                            v_map_res[max_v][max_u] = v_map[min_tv][min_tu]
                            # part_id[max_v][max_u][0] = car_id
                            # part_id[max_v][max_u][1] = 50
                            # part_id[max_v][max_u][2] = part_index
                            part_id[max_v][max_u][0] = color_table[part_index][2]
                            part_id[max_v][max_u][1] = color_table[part_index][1]
                            part_id[max_v][max_u][2] = color_table[part_index][0]

                            depth_map[max_v][max_u] = des
                    continue
                # colors = get_texture_part_color()
                append_mask = des < depth_map[min_v:max_v, min_u:max_u]
                depth_map[min_v:max_v, min_u:max_u][append_mask] = des
                u_map_res[min_v:max_v, min_u:max_u][append_mask] = \
                cv2.resize(u_map[min_tv:max_tv, min_tu:max_tu], (max_u - min_u, max_v - min_v))[append_mask]
                v_map_res[min_v:max_v, min_u:max_u][append_mask] = \
                cv2.resize(v_map[min_tv:max_tv, min_tu:max_tu], (max_u - min_u, max_v - min_v))[append_mask]
                # part_id[min_v:max_v, min_u:max_u, 0:1][append_mask] = car_id
                # part_id[min_v:max_v, min_u:max_u, 1:2][append_mask] = 50
                # part_id[min_v:max_v, min_u:max_u, 2:3][append_mask] = part_index
                part_id[min_v:max_v, min_u:max_u, 0:1][append_mask] = color_table[part_index][2]
                part_id[min_v:max_v, min_u:max_u, 1:2][append_mask] = color_table[part_index][1]
                part_id[min_v:max_v, min_u:max_u, 2:3][append_mask] = color_table[part_index][0]
            # img = append_uv_map(u_map_res, img, alpha=0.5)

            # cv2.imwrite('result.jpg', img)

        # u_map_res = np.uint8(u_map_res * 255)
        # np.save(uv_ouput_dir + pose_file.split('.')[0] + '_u_map', u_map_res)
        # v_map_res = np.uint8(v_map_res * 255)
        # np.save(uv_ouput_dir + pose_file.split('.')[0] + '_v_map', v_map_res)
        cv2.imwrite(uv_ouput_dir + pose_file.split('.')[0] + '_part_id.png', part_id)
        ori = cv2.imread('/home/beta/SG2020/densepose_depth_v2/test_image/' + pose_file.split('.')[0] + '.png')
        ori = ori * 0.7 + part_id * 0.3
        ori = ori.astype(np.uint8)
        cv2.imwrite(uv_ouput_dir + pose_file.split('.')[0] + '.png', ori)

        # np.save(uv_ouput_dir + pose_file.split('.')[0] + '_depth', depth_map)


if __name__ == "__main__":
    pose_dir = '/home/beta/SG2020/densepose_depth_v2/pose_gt/'
    model_dir = '/home/beta/SG2020/densepose/Kitti_models/'
    texture_map = '/home/beta/SG2020/densepose/Kitti_models/Template18_new.PNG'
    # raw_image_dir = '/media/vrlab/556cb30b-ef30-4e11-a77a-0e33ba901842/Auto-Car-Data/c5/'
    uv_ouput_dir = '/home/beta/SG2020/kitti_3d/test_vis/'
    # cur_files = [f.split('_')[0] for f in os.listdir('./labeled/depth/')]
    # result_files = [f.split('_')[0] for f in os.listdir('/media/beta/Newsmy/Kitti_3D/kitti_3d/uv_part_result') if 'part' in f]
    pose_files = [f for f in os.listdir(pose_dir)]

    fx = 721.5377
    fy = 721.5377
    cx = 609.55
    cy = 172.854
    camera_mat = np.array([[fx, 0, cx],
                           [0, fy, cy],
                           [0, 0, 1]])
    part_bboxes = get_part_patch_box(texture_map)
    u_map, v_map = convert_texture_map_to_uv_map(part_bboxes, texture_map)

    pcs, face_index, t_u, t_v = read_pcs(model_dir)

    num_of_worker = 12
    num_per_worker = len(pose_files) // num_of_worker
    processes = []
    for i in range(num_of_worker):
        if i == num_of_worker - 1:
            p = Process(target=uv_ge, args=(
            pose_files[i * num_per_worker:], uv_ouput_dir, pose_dir, model_dir, camera_mat, u_map, v_map, pcs,
            face_index, t_u, t_v, part_bboxes))
        else:
            p = Process(target=uv_ge, args=(
            pose_files[i * num_per_worker:(i + 1) * num_per_worker], uv_ouput_dir, pose_dir, model_dir, camera_mat,
            u_map, v_map, pcs, face_index, t_u, t_v, part_bboxes))
        p.start()
        processes.append(p)

