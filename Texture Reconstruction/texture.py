import numpy as np
import cv2
import math
import time
import os

def get_uv_index(filename):
    with open(filename) as f:
        line = f.readline()
        u = []
        v = []
        vertex_index = []
        while line:
            items = line.split(' ')
            u.append(int(items[1]))
            v.append(int(items[2]))
            vertex_index.append(int(items[3].split('\n')[0]))
            line = f.readline()

    return u, v, vertex_index

def read_pose_file(path):
    poses = []
    with open(path) as f:
        for line in f:
            items = line.split()
            pose = [float(x) for x in items] 
            poses.append(pose)
    return poses

def read_model(point_cloud_path):
    x = []
    y = []
    z = []
    u = []
    v = []
    other_parts = []
    with open(point_cloud_path) as f:
        line = f.readline()

        while line:
            if line[0] == 'v' and line[1] == ' ':
                items = line.split(' ')
                x.append(float(items[1]))
                y.append(float(items[2]))
                z.append(float(items[3].split('\n')[0]))

            elif line[0] == 'v' and line[1] == 't':
                items = line.split(' ')
                u.append(int((float(items[1])) * 2048))
                v.append(int((1 - float(items[2])) * 2048))
                other_parts.append(line)
            elif line[0] != 'm':
                other_parts.append(line)               
            line = f.readline()

    return np.array([x, y, z]), np.array(u), np.array(v), other_parts


def read_face_index(model_path, index=0):
    with open(model_path) as f:
        line = f.readline()
        face_index = []
        while (line):
            if line[0] == 'f':
                item = line.split(' ')
                f1 = int(item[1].split('/')[0]) - 1
                f2 = int(item[2].split('/')[0]) - 1
                f3 = int(item[3].split('/')[0]) - 1

                face_index.append([f1, f2, f3])

            line = f.readline()

    return np.array(face_index)


def write_pointcloud_np(write_dir, vertexs_np, other_parts, car_id):
    with open(write_dir + str(car_id) + '.obj', 'w') as f:
        f.write('mtllib ' + str(car_id) + '.mtl\n')
        for k, i in enumerate(range(0, vertexs_np.shape[1])):
            f.write(
                'v ' + str(vertexs_np[0][i]) + ' ' + str(vertexs_np[1][i]) + ' ' + str(vertexs_np[2][i]) + '\n')

        for l in other_parts:
            f.write(l)
    with open(write_dir + str(car_id) + '.mtl', 'w') as f:
        # newmtl default
        # Ns 10.0000
        # Ni 1.5000
        # d 1.0000
        # Tr 0.0000
        # Tf 1.0000 1.0000 1.0000 
        # illum 2
        # Ka 0.5880 0.5880 0.5880
        # Kd 0.5880 0.5880 0.5880
        # Ks 0.0000 0.0000 0.0000
        # Ke 0.0000 0.0000 0.0000
        # map_Ka Template18_new2.png
        # map_Kd Template18_new2.png
        f.write('newmtl default\n')
        f.write('Ns 10.0000\n')
        f.write('Ni 1.5000\n')
        f.write('d 1.0000\n')
        f.write('Tr 0.0000\n')
        f.write('Tf 1.0000 1.0000 1.0000 \n')
        f.write('illum 2\n')
        f.write('Ka 0.5880 0.5880 0.5880\n')
        f.write('Kd 0.5880 0.5880 0.5880\n')
        f.write('Ks 0.0000 0.0000 0.0000\n')
        f.write('Ke 0.0000 0.0000 0.0000\n')
        f.write('map_Ka Template18_new2_' + str(car_id) + '.png\n')
        f.write('map_Kd Template18_new2_' + str(car_id) + '.png\n')



def get_rotation_mat(a, b, c):
    rotation = np.zeros((3, 3))
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


def get_texture_part_color():
    part1 = [100, 111, 156]
    part2 = [5, 28, 126]
    part3 = [188, 110, 83]
    part4 = [75, 13, 159]
    part5 = [8, 78, 183]
    part6 = [216, 166, 255]
    part7 = [113, 165, 0]
    part8 = [229, 114, 84]
    part9 = [140, 60, 39]
    part10 = [112, 252, 57]

    part11 = [247, 77, 149]
    part12 = [32, 148, 9]
    part13 = [166, 104, 6]
    part14 = [7, 212, 133]
    part15 = [1, 251, 1]
    part16 = [2, 2, 188]
    part17 = [219, 251, 1]
    part18 = [96, 94, 92]

    return [part1, part2, part3, part4, part5, part6, part7, part8, part9, part10, part11, part12, part13, part14, part15, part16, part17, part18]

def get_unvisiable_part(part_img, label_img, label_index):
    # Get the car
    mask = (label_img == label_index)
    roi = part_img[:, :, 0][mask]
    parts_color = get_texture_part_color()
    unvisiable_part = []
    visiable_part = []
    fill_flag = []
    for i, part_color in enumerate(parts_color):
        search = np.argwhere(roi == part_color[2])
        if search.shape[0] < 20:
           print(i)
           unvisiable_part.append(i)
           fill_flag.append(False)
        else:
           visiable_part.append(i)
           fill_flag.append(True)

    return unvisiable_part, visiable_part, fill_flag

def get_main_color(main_color_index, parts_color, texture_map, ori_texure_map):
    main_rgb = []
    if main_color_index is not None:
        mask = (ori_texure_map[:, :, 2] == parts_color[main_color_index][0]) & (texture_map[:, :, 1] != 0)
        rhist, rbin_edges = np.histogram((texture_map[:, :, 2][mask]), bins=25)

        r = rbin_edges[np.argmax(rhist)]
        ghist, gbin_edges = np.histogram(texture_map[:, :, 1][mask], bins=25)
        g = gbin_edges[np.argmax(ghist)]
        bhist, bbin_edges = np.histogram(texture_map[:, :, 0][mask], bins=25)
        b = bbin_edges[np.argmax(bhist)]

        main_rgb = [r, g, b]
    else:
        main_rgb = [70, 70, 70]
    return main_rgb

def get_main_color_patch(main_color_index, parts_color, texture_map, ori_texure_map):
    if main_color_index is not None:
        roi_uv = np.argwhere((ori_texure_map[:, :, 2] == parts_color[main_color_index][0]) & (texture_map[:, :, 1] != 0))
        min_u = np.min(roi_uv[:, 0])
        min_v = np.min(roi_uv[:, 1])
        max_u = np.max(roi_uv[:, 0])
        max_v = np.max(roi_uv[:, 1])
        du = max_u - min_u
        dv = max_v - min_v
        min_u += du//5
        max_u -= du//5
        min_v += dv//5
        max_v -= dv//5
        patch = texture_map[min_u: max_u, min_v: max_v, :]
        patch = cv2.resize(patch, (2048, 2048))

        return patch
    else:
        return None

# def fill_in_patch(patch, mask, texture_map, pre_define_texture):
#     roi_uv = np.argwhere((ori_texure_map[:, :, 2] == parts_color[main_color_index][0]) & (texture_map[:, :, 1] != 0))
#     min_u = np.min(roi_uv[:, 0])
#     min_v = np.min(roi_uv[:, 1])
#     max_u = np.max(roi_uv[:, 0])
#     max_v = np.max(roi_uv[:, 1])

def remove_unvisiable_area(texture_map,part_img, ori_texure_map, label_img, label_index, pre_define_texture):
    unvisiable_part, visiable_part, fill_flag = get_unvisiable_part(part_img, label_img, label_index)
    # unvisiable_part = [0, 1, 3, 5, 7, 8, 9, 13, 14, 15, 16, 17]
    # visiable_part = [2, 4, 6, 10, 11, 12]
    # fill_flag = [False, False, True, False, True, False, True, False, False, False, True, True, True, False, False, False, False, False]
    parts_color = get_texture_part_color()
    for part_id in unvisiable_part:
        mask = (ori_texure_map[:, :, 2] == parts_color[part_id][0])
        b = texture_map[:, :, 0]
        g = texture_map[:, :, 1]
        r = texture_map[:, :, 2]
        b[mask] = 0
        g[mask] = 0
        r[mask] = 0

    main_color_index = None
    for index in [6, 7, 11, 15]:
        if fill_flag[index] == True:
            main_color_index = index
            break
    main_rgb = get_main_color(main_color_index, parts_color, texture_map, ori_texure_map)
    # write_with_alpha('texture\\' + '180114_030534471_Camera_5' + '\\process0.png', texture_map)

    blend_and_fill_black_area(texture_map, ori_texure_map, parts_color, visiable_part)

    # write_with_alpha('texture\\' + '180114_030534471_Camera_5' + '\\process1.png', texture_map)

    main_patch = get_main_color_patch(main_color_index, parts_color, texture_map, ori_texure_map)
    for part_id in unvisiable_part:
        mask = (ori_texure_map[:, :, 2] == parts_color[part_id][0])
        if part_id < 10:
            symmetry_id = None
            if part_id % 2 == 0:
                for p in visiable_part:
                    if p == part_id + 1:
                        symmetry_id = p
            else:
                for p in visiable_part:
                    if p == part_id - 1:
                        symmetry_id = p
            if symmetry_id is not None:
                symmetry_part(texture_map, ori_texure_map, parts_color, part_id, symmetry_id)
                if symmetry_id == 6 or symmetry_id == 7:
                    refine_black_area(texture_map, ori_texure_map, parts_color, part_id)
                fill_flag[part_id] = True
            else:
                b = texture_map[:, :, 0]
                g = texture_map[:, :, 1]
                r = texture_map[:, :, 2]
                b[mask] = 0
                g[mask] = 0
                r[mask] = 0
        else:
            b = texture_map[:, :, 0]
            g = texture_map[:, :, 1]
            r = texture_map[:, :, 2]
            b[mask] = 0
            g[mask] = 0
            r[mask] = 0
    #cv2.imwrite('texture\\' + '180116_040122304_Camera_5' + '\\process2.png', texture_map)
   #  write_with_alpha('texture\\' + '180114_030534471_Camera_5' + '\\process2.png', texture_map)
    # fill the wheels
    for index in [0 ,1, 8, 9]:
        mask = (ori_texure_map[:, :, 2] == parts_color[index][0])
        texture_map[mask] = pre_define_texture[mask]
        fill_flag[index] = True

    for i, flag in enumerate(fill_flag):
        if flag and i != 17:
            continue
        mask = (ori_texure_map[:, :, 2] == parts_color[i][0])
        if i == 4 or i == 5 or i == 2 or i == 3:
            mask1 = mask & (pre_define_texture[:, :, 2] > 100) & (pre_define_texture[:, :, 2] < 200) & (pre_define_texture[:, :, 1] < 20)
            texture_map[mask] = pre_define_texture[mask]
            if main_patch is None:
                texture_map[:, :, 2][mask1] = main_rgb[0]
                texture_map[:, :, 1][mask1] = main_rgb[1]
                texture_map[:, :, 0][mask1] = main_rgb[2]
            else:
                texture_map[mask1] = main_patch[mask1]
        if i == 12 or i == 14:
            if i == 12:
                index = 14
            else:
                index = 12
            if fill_flag[index] == True:
                rgb = get_main_color(index, parts_color, texture_map, ori_texure_map)
                texture_map[:, :, 2][mask] = rgb[0]
                texture_map[:, :, 1][mask] = rgb[1]
                texture_map[:, :, 0][mask] = rgb[2]
            else:
                if fill_flag[4] == True:
                    index = 4
                else:
                    index = 5
                rgb = get_main_color(index, parts_color, texture_map, ori_texure_map)
                texture_map[:, :, 2][mask] = rgb[0]
                texture_map[:, :, 1][mask] = rgb[1]
                texture_map[:, :, 0][mask] = rgb[2]

        if i == 10 or i == 15:
            mask1 = mask & (pre_define_texture[:, :, 2] > 50) & (pre_define_texture[:, :, 2] < 255) & (pre_define_texture[:, :, 1] < 20)
            texture_map[mask] = pre_define_texture[mask]
            if main_patch is None:
                texture_map[:, :, 2][mask1] = main_rgb[0]
                texture_map[:, :, 1][mask1] = main_rgb[1]
                texture_map[:, :, 0][mask1] = main_rgb[2]
            else:
                texture_map[mask1] = main_patch[mask1]

        if i == 16 or i == 13 or i == 6 or i == 7 or i == 11:
            if main_patch is None:

                texture_map[:, :, 2][mask] = main_rgb[0]
                texture_map[:, :, 1][mask] = main_rgb[1]
                texture_map[:, :, 0][mask] = main_rgb[2]
            else:
                texture_map[mask] = main_patch[mask]

        if i == 17:
            texture_map[:, :, 2][mask] = 15
            texture_map[:, :, 1][mask] = 15
            texture_map[:, :, 0][mask] = 15

def blend_and_fill_black_area(texture_map, ori_texure_map, parts_color, visiable_part, iteration = 6, block_size = 1):
    #
    for vis_part_id in visiable_part:
        for _ in range(iteration):
            black_area_uv = np.argwhere(
                (ori_texure_map[:, :, 2] == parts_color[vis_part_id][0]) & (texture_map[:, :, 1] == 0))
            for i in range(black_area_uv.shape[0]):
                u = black_area_uv[i][0]
                v = black_area_uv[i][1]
                block = texture_map[u - block_size:u + block_size, v - block_size:v + block_size, :]
                count = np.sum((block[:, :, 1] > 0))
                if count == 0:
                    continue
                texture_map[u][v][0] = np.sum(block[:, :, 0]) / count
                texture_map[u][v][1] = np.sum(block[:, :, 1]) / count
                texture_map[u][v][2] = np.sum(block[:, :, 2]) / count

def refine_black_area(texture_map, ori_texure_map, parts_color, visiable_part):
    #

        black_area_uv = np.argwhere(
                (ori_texure_map[:, :, 2] == parts_color[visiable_part][0]) & (texture_map[:, :, 1] == 0))

        block_size = 20

        for i in range(black_area_uv.shape[0]):
            u = black_area_uv[i][0]
            v = black_area_uv[i][1]
            block = texture_map[u - block_size:u + block_size, v - block_size:v + block_size, :]
            count = np.sum((block[:, :, 1] > 0))
            if count == 0:
                continue
            texture_map[u][v][0] = np.sum(block[:, :, 0]) / count
            texture_map[u][v][1] = np.sum(block[:, :, 1]) / count
            texture_map[u][v][2] = np.sum(block[:, :, 2]) / count

def symmetry_part(texture_map, ori_texure_map, parts_color, part_id, symmetry_id):
    symmetric_axis = 455 + 463/2
    uv_index = np.argwhere(ori_texure_map[:, :, 0] == parts_color[part_id][2])
    symmetry_part_index = np.copy(uv_index)
    symmetry_part_index[:, 1] = uv_index[:, 1] + 2 * (symmetric_axis - uv_index[:, 1])
    for i in range(uv_index.shape[0]):
        texture_map[uv_index[i][0]][uv_index[i][1]][0] = texture_map[symmetry_part_index[i][0]][symmetry_part_index[i][1]][0]
        texture_map[uv_index[i][0]][uv_index[i][1]][1] = texture_map[symmetry_part_index[i][0]][symmetry_part_index[i][1]][1]
        texture_map[uv_index[i][0]][uv_index[i][1]][2] = texture_map[symmetry_part_index[i][0]][symmetry_part_index[i][1]][2]




if __name__ == '__main__':
    # USA_Data()
    fx = 721.5377
    fy = 721.5377
    cx = 609.55
    cy = 172.854
    camera_mat = np.array([[fx, 0, cx],
                            [0, fy, cy],
                            [0, 0, 1]])
    img_name = '000010'
    obj_dir = './new_kitti_model/'
    face_index = read_face_index(obj_dir + 'SUV_BD269_Volkswagen_Tiguan_2017.obj')
    img = cv2.imread('./' + img_name + '.png')
    label_img = cv2.imread('./' + img_name + '_part_id.png')[:,:,0]
    part_img = cv2.imread('./' + img_name + '_part.png')
    ori_texure_map = cv2.imread(obj_dir + 'Template18_new.PNG')
    pre_define_texture = cv2.imread('11.png')
    poses = read_pose_file('./' + img_name + '.txt')
    car_id = 1

    car_names = [f for f in os.listdir(obj_dir) if 'obj' in f]
    car_names.sort()
    if not os.path.exists('./recon_texture/' + img_name):
        os.makedirs('./recon_texture/' + img_name)

    # for car_id in range(len(poses)):
    pc, t_u, t_v, other_parts = read_model(obj_dir + car_names[int(poses[car_id][0])])

    print('car_id = ', car_id)
    pose = poses[car_id]
    rot_mat = get_rotation_mat(pose[4] + math.pi/2, pose[5], pose[6])
        
    pc2 = np.dot(rot_mat, pc)
    pc2[0, :] += pose[1]
    pc2[1, :] += pose[2]
    pc2[2, :] += pose[3]
    write_pointcloud_np('recon_texture/' + img_name + '/', pc2, other_parts, car_id)
    pp = np.dot(camera_mat, pc2)
    v = np.int32(pp[0, :] / pp[2, :])
    u = np.int32(pp[1, :] / pp[2, :])
    uv = np.vstack((u, v)).T
           

    patch = uv[face_index.T]
    patch = np.transpose(patch, (0, 2, 1))

    t_uv = np.vstack((t_v, t_u)).T
    t_patch = t_uv[face_index.T]
    t_patch = np.transpose(t_patch, (0, 2, 1))

    faca_num = patch.shape[2]

    texture_map = np.zeros((2048, 2048, 3))

    for i in range(faca_num):
        max_uv = np.max(patch[:, :, i], axis=0)
        min_uv = np.min(patch[:, :, i], axis=0)
        max_t_uv = np.max(t_patch[:, :, i], axis=0)
        min_t_uv = np.min(t_patch[:, :, i], axis=0)

        patch_texture = texture_map[min_t_uv[0]:max_t_uv[0], min_t_uv[1]:max_t_uv[1]]
        patch_img = img[min_uv[0]:max_uv[0], min_uv[1]:max_uv[1]]
        if patch_img.shape[0] == 0 or patch_img.shape[1] == 0 or patch_texture.shape[0] == 0 or patch_texture.shape[1] == 0 or patch_texture.shape[0] * patch_texture.shape[1] > 100:
            continue
        texture_map[min_t_uv[0]:max_t_uv[0], min_t_uv[1]:max_t_uv[1]] = cv2.resize(patch_img, (
        patch_texture.shape[1], patch_texture.shape[0]))


    remove_unvisiable_area(texture_map, part_img, ori_texure_map, label_img, car_id, pre_define_texture)

    cv2.imwrite('recon_texture/' + img_name + '/Template18_new2_' + str(car_id) + '.png', texture_map)
    
