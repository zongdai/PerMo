import cv2
import numpy as np
from .Car_3DMM import read_mapper_file, show_mapper, get_point3d2, get_projection_matrix,get_model_bounding_box,project_bounding2img, caculate_error
from .image_editing import read_pose_file, get_car_name, read_pointcloud, get_rotation_mat, get_car_index
import os
import random
def read_model(point_cloud_path, read_face = False, read_color = False, read_vt = False, scale = 1.0):
    x = []
    y = []
    z = []
    u = []
    v = []
    rgb = []
    face_index = []
    other_strs = []
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
                other_strs.append(line)
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
                other_strs.append(line)
                items = line.split(' ')
                u.append(int((float(items[1])) * 2048))
                v.append(int((1 - float(items[2])) * 2048))
            line = f.readline()

    return np.array([x, y, z]), np.array(face_index), np.array(u), np.array(v), np.array(rgb).T, other_strs
def get_texture_part_color():
    color_table = [
		[247, 77, 149],
		[32, 148, 9],
		[166 ,104, 6],
		[7 ,212, 133],
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
def get_part_patch_box(texture_map_path):
    texture_map = cv2.imread(texture_map_path)
    parts_color = get_texture_part_color()
    bboxes = []
    for i, color in enumerate(parts_color):
        area_uv = np.argwhere(texture_map[:, :, 2] == color[0])
        # print(area_uv.shape)
        min_v = np.min(area_uv[:, 0])
        min_u = np.min(area_uv[:, 1])
        max_v = np.max(area_uv[:, 0])
        max_u = np.max(area_uv[:, 1])
        cv2.putText(texture_map, str(int(i)), (int(min_u), int(min_v)), 1, 5, (0, 0, 255) ,1)

        bboxes.append([min_v, min_u, max_v, max_u]) # y, x, height, width
    cv2.imwrite('./part.png', texture_map)
    return bboxes

def get_color():
	color_table = [
		[28, 125, 134],
		[79, 126, 0],
		[189, 255, 0],
		[0, 255, 255],
		[0, 128, 255],
		[255, 0, 147],
		[0, 36, 98],
		[0, 0, 255],
		[87, 37, 1],
		[0, 255, 0],
		[255, 0, 17],
		[132, 125, 2],
		[132, 0, 255],
		[255, 255, 0],
	]

	return color_table[random.randint(0, len(color_table)-1)]

def flann_match_keypoints(part_uv, target_uv, texture):
    match_vertexs_index = []

        
    index_params = dict(algorithm=0, trees = 15)
    search_params = dict(checks=30)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
        
    matches = flann.knnMatch(np.float32(part_uv), np.float32(target_uv), k=1)

    for m in matches:
        vertex_index = (m[0].trainIdx)
        match_vertexs_index.append(vertex_index)
        # cv2.circle(texture,(target_uv[vertex_index][0],target_uv[vertex_index][1]), 2, (0,0,255), -1)
        # print(n.distance)
    # cv2.imwrite('tt.png', texture)
    return match_vertexs_index

def caculate_iou(re_po_box, target_box, img_shape):
    bg1 = np.zeros(img_shape, dtype=np.int8)
    bg2 = np.zeros(img_shape, dtype=np.int8)
    bg1[int(re_po_box[1]): int(re_po_box[3]), int(re_po_box[0]):int(re_po_box[2])] = 1
    bg2[int(target_box[1]): int(target_box[3]), int(target_box[0]):int(target_box[2])] = 1
    mask1_binary = bg1 == 1
    mask2_binary = bg2 == 1
    intersection = (mask1_binary & mask2_binary)
    union = (mask1_binary | mask2_binary)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def estimate_6Dof_pose_multiprocess(vertexs_index_arr, uv_arr, bbox_arr, img, pcs, img_name, car_names):
    for vertexs_index, uv, bbox in zip(vertexs_index_arr, uv_arr, bbox_arr):
        estimate_6Dof_pose(vertexs_index, uv, img, pcs, car_names, bbox, img_name)

def get_camera_mat(name, calib_dir):
    camera_mat = np.array([[7.215377000000e+02, 0, 6.095593000000e+02],
                        [0, 7.215377000000e+02, 1.728540000000e+02],
                        [0, 0 , 1 ]])
    with open( os.path.join(calib_dir, name.split('.')[0] + '.txt')) as f:
        line = f.readline()
        item = line.split()
        fx, cx, fy, cy = float(item[1]), float(item[3]), float(item[6]), float(item[7])
        camera_mat = np.array([[fx, 0, cx],
                            [0, fy, cy],
                            [0, 0 , 1 ]])
    return camera_mat

def estimate_6Dof_pose(vertexs_index, uv, img, pcs, car_names, bbox, img_name):
    '''
    vertexs_index : 3D vertex indexs
    uv            : corresponding 2D pix in image
    img           : img matrix to vis
    pcs           : tamplates models point cloud
    car_names     : tamplates models name
    bbox          : 2D detection result
    img_name      : input image name
    '''
    
    
    camera_mat = get_camera_mat(img_name)
    
    max_iou = 0
    reprojectionError = 3
    min_error = 1e9
    for pc, car_name in zip(pcs, car_names):
        # get 3D points by vertex index
        point_3d1 = get_point3d2(vertexs_index, pc)
        # for current tamplate, using ransac_pnp get pose
        projection_matrix, rt, pose = get_projection_matrix(camera_mat, uv, point_3d1, reprojectionError)
        # caculate re-projection error
        error = caculate_error(point_3d1.T, uv[:, 0], uv[:, 1], projection_matrix)

        # caculate iou between 2D bbox and re-projection bbox by current model
        E = np.ones((1, pc.shape[1]))
        new_point_3d = np.vstack((pc, E))
        new_point_3d = np.dot(projection_matrix, new_point_3d)
        u = new_point_3d[0,:] / new_point_3d[2,:]
        v = new_point_3d[1,:] / new_point_3d[2,:]
        u_min = max(np.min(u), 0)
        v_min = max(np.min(v), 0)
        u_max = min(np.max(u), img.shape[1])
        v_max = min(np.max(v), img.shape[0])
        iou = caculate_iou([u_min, v_min, u_max, v_max], bbox, img.shape)

        # get the max iou shape for last result
        if iou > max_iou:
            best_projection_matrix = projection_matrix
            best_rt = rt
            best_pose = pose
            min_error = error
            max_iou = iou
            best_shape = pc
            best_shape_index = get_car_index(car_name)
    
    ########## vis poseres ############
    E = np.ones((1, best_shape.shape[1]))
    new_point_3d = np.vstack((best_shape, E))
    new_point_3d = np.dot(best_projection_matrix, new_point_3d)
    u = new_point_3d[0,:] / new_point_3d[2,:]
    v = new_point_3d[1,:] / new_point_3d[2,:]
    color = get_color()
    for i, j in zip(u, v):
        if random.randint(0,100) < 95:
            continue
        if i < 0 or i >= img.shape[1] or j < 0 or j >= img.shape[0]:
            continue
        img[int(j)][int(i)][0] = color[2]
        img[int(j)][int(i)][1] = color[1]
        img[int(j)][int(i)][2] = color[0]
    cv2.imwrite('./apollo_test/back_pro/' + img_name, img)
    ##########  end vis poseres ############


    ### store the result ##
    with open('./apollo_test/coarse_pose_remove_part145/' + img_name.split('.')[0] + '.txt', 'a') as f:
        f.write(str(best_shape_index) + ' ' + str(best_pose[0]) + ' ' + str(best_pose[1]) + ' ' + str(best_pose[2]) + ' ' + str(best_pose[3]) + ' ' + str(best_pose[4]) + ' ' + str(best_pose[5]) + '\n')


def estimate_6Dof_pose_with_partiou(vertexs_index, uv, img, pcs, car_names, bbox, img_name, spcs, face_indexs, t_us, t_vs, part_seg, part_bboxs, camera_mat):
    '''
    vertexs_index : 3D vertex indexs
    uv            : corresponding 2D pix in image
    img           : img matrix to vis
    pcs           : tamplates models point cloud
    car_names     : tamplates models name
    bbox          : 2D detection result
    img_name      : input image name
    '''
    
    
    
    max_iou = 0
    reprojectionError = 3
    min_error = 1e9
    best_mask = None
    for spc, pc, car_name , face_index, t_u, t_v in zip(spcs, pcs, car_names, face_indexs, t_us, t_vs):
        # get 3D points by vertex index
        point_3d1 = get_point3d2(vertexs_index, pc)
        # for current tamplate, using ransac_pnp get pose
        projection_matrix, rt, pose = get_projection_matrix(camera_mat, uv, point_3d1, reprojectionError)
        # caculate re-projection error
        error = caculate_error(point_3d1.T, uv[:, 0], uv[:, 1], projection_matrix)

        # caculate iou between 2D bbox and re-projection bbox by current model
        [a, b, c, x, y, z] = pose
        res, mask = render(spc, face_index, t_u, t_v, x, y, z, a, b, c, camera_mat, img.shape[1], img.shape[0], part_bboxs)
        iou_score = get_part_iou(part_seg, res[:, :, 0])
        # get the max iou shape for last result
        if iou_score > max_iou:
            best_projection_matrix = projection_matrix
            best_rt = rt
            best_pose = pose
            min_error = error
            max_iou = iou_score
            best_shape = pc
            best_shape_index = (car_name)
            best_mask = res[:, :, 0]
    ########## vis poseres ############
    
    ##########  end vis poseres ############

    ### store the result ##
    vis_part(best_mask, img)
    
    width = np.max(best_shape[2, :]) - np.min(best_shape[2, :])
    height = np.max(best_shape[1, :]) - np.min(best_shape[1, :])
    length = np.max(best_shape[0, :]) - np.min(best_shape[0, :])
    [a, b, c, x, y, z] = best_pose
    bbox3d = get_model_bounding_box(best_shape)
    project_bounding2img(img, bbox3d, best_projection_matrix)

    best_shape = np.vstack((best_shape, np.ones((1, best_shape.shape[1]))))
    best_shape = np.dot(best_rt, best_shape)
    detection3d_res = [bbox[0], bbox[1], bbox[2], bbox[3], height, width, length, x, y, z, b, 0.99]
    return img, detection3d_res, best_shape
            

def papare_for_reconstruction(model_dir, texture_path):
    import math
    # pc, face_index, t_u, t_v, _ = read_model('./recon/reconstruction/01.obj', read_face=True, read_vt=True)
    pc, face_index, t_u, t_v, _, others_strs = read_model(os.path.join(model_dir, 'Coupe_BD331_BMW_Z4.obj'), read_face=True, read_vt=True)
    bboxes = get_part_patch_box(texture_path)
    
    return bboxes, np.hstack((t_u.reshape(-1,1), t_v.reshape(-1,1))), others_strs

def convert_part_uv_to_global_uv(u_map, v_map, part_mask_img, bboxes):
    part_indexs = np.unique(part_mask_img)[1:]
    # print(part_indexs)
    parts_uv = []
    uv_in_raw = []
    
    for part_index in part_indexs:
        if part_index in [8]:
            continue
        part_uv_in_raw = np.argwhere(part_mask_img == part_index)
        part_mask = part_mask_img == part_index
        
       
        
        u = u_map[part_uv_in_raw[:,0], part_uv_in_raw[:,1]].reshape(-1, 1)
        v = v_map[part_uv_in_raw[:,0], part_uv_in_raw[:,1]].reshape(-1, 1)
        
        bbox = bboxes[part_index-1]
        u = bbox[1] + u*(bbox[3] - bbox[1])
        v = bbox[0] + v*(bbox[2] - bbox[0])
        uv = np.hstack((u, v))
        parts_uv.append(uv)
        part_uv_in_raw[:, [0,1]] = part_uv_in_raw[:, [1,0]]
        uv_in_raw.append(part_uv_in_raw)
    return np.vstack(parts_uv), np.vstack(uv_in_raw)

def get_pc_from_depth(part_uv_in_raw, camera_mat, depth_map):
    fy = fx = camera_mat[0][0]
    cx = camera_mat[0][2]
    cy = camera_mat[1][2]
    
    xx = ((part_uv_in_raw[:,0] - cx) * depth_map[part_uv_in_raw[:,1], part_uv_in_raw[:,0]] / fx).reshape(-1, 1)
    yy = ((part_uv_in_raw[:,1] + 1 - cy) * depth_map[part_uv_in_raw[:,1], part_uv_in_raw[:,0]] / fy).reshape(-1, 1)
    zz = (depth_map[part_uv_in_raw[:,1], part_uv_in_raw[:,0]]).reshape(-1, 1)
    target_3d = np.hstack((xx, yy, zz)).T
    return target_3d
def convert_part_uv_to_global_uv_sparse(res_sparce, bboxes):
    u = []
    v = []
    c_u = []
    c_v = []
    for item in res_sparce:
        bbox = bboxes[item['part']-1]
        u.append(bbox[1] + item['u']*(bbox[3] - bbox[1]))
        v.append(bbox[0] + item['v']*(bbox[2] - bbox[0]))
        c_u.append(item['c_u'])
        c_v.append(item['c_v'])
    u = np.array(u).reshape(-1, 1)
    v = np.array(v).reshape(-1, 1)
    c_u = np.array(c_u).reshape(-1, 1)
    c_v = np.array(c_v).reshape(-1, 1)

    return [np.hstack((u, v))], np.hstack((c_u, c_v))
# flann_match_keypoints_test()

def papare_sim_models(model_dir):
    model_names = [f for f in os.listdir(model_dir) if 'obj' in f]
    model_names.sort()
    pcs, face_indexs, t_us, t_vs = [], [], [], []
    import math
    for name in model_names:
        pc, face_index, t_u, t_v, _, _ = read_model(os.path.join(model_dir, name), read_face=True, read_vt=True)
        pc = np.dot(get_rotation_mat(math.pi/2, 0, 0), pc)
        # pc = np.dot(get_rotation_mat(0, -math.pi/2, 0), pc)
        pcs.append(pc)
        face_indexs.append(face_index)
        t_us.append(t_u)
        t_vs.append(t_v)
    return pcs, face_indexs, t_us, t_vs

def papare_template_vertexs(model_dir):
    model_names = [f for f in os.listdir(model_dir) if 'obj' in f]
    model_names.sort()
    templates = []
    car_names = []
    import math
    for name in model_names:
        pc = read_pointcloud(os.path.join(model_dir, name))
        pc = np.dot(get_rotation_mat(math.pi/2, 0, 0), pc)
        # pc = np.dot(get_rotation_mat(0, -math.pi/2, 0), pc)
        templates.append(pc)
        car_names.append(name.split('.')[0])
    return templates, car_names


def vis_part(part_mask, img):
    colors = get_texture_part_color()
    for i in range(1, 19):
        append_area = part_mask == i
        img[:, :, 0][append_area] = img[:, :, 0][append_area]*0.7 + colors[i-1][2] * 0.3
        img[:, :, 1][append_area] = img[:, :, 1][append_area]*0.7 + colors[i-1][1] * 0.3
        img[:, :, 2][append_area] = img[:, :, 2][append_area]* 0.7 + colors[i-1][0] * 0.3

def read_pose_file(pose_file):

    with open(pose_file) as f:
        car_infos = []
        line = f.readline()
        while(line):
            info = {}
            items = line.split(' ')
            # print(items)
            info["name"] = items[0]
            info["a"] = float(items[4])
            info["b"] = float(items[5])
            info["c"] = float(items[6].split('\n')[0])
            info["tx"] = float(items[1])
            info["ty"] = float(items[2])
            info["tz"] = float(items[3])
            if float(items[3]) == 0:
                line = f.readline()
                continue
            # print(info)
            car_infos.append(info)
            line = f.readline()
    return car_infos

def get_part_iou(pre_part_img, model_part):
    weight = [1 for i in range(18)] 
    part_ids = np.unique(pre_part_img)
    iou_score = 0
    for part_id in part_ids:
        if part_id == 0 or part_id == 8:
            continue
        pre_mask = np.zeros(pre_part_img.shape, dtype=np.bool)
        model_mask = np.zeros(pre_part_img.shape , dtype=np.bool)
        pre_mask[pre_part_img == part_id] = True
        model_mask[model_part == part_id] = True
        iou = caculate_mask_iou(pre_mask, model_mask)
        iou_score += iou * weight[int(part_id-1)]
    return iou_score

def caculate_mask_iou(mask1_binary, mask2_binary):
	
	intersection = (mask1_binary & mask2_binary)
	union = (mask1_binary | mask2_binary)
	iou = np.sum(intersection) / np.sum(union)
	return iou

def render(pc, face_index, t_u, t_v, x, y, z, a, b, c, camera_mat, width, height, part_bboxs, is_render_part=True):

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
    # print(face_index)
    tem_depth = np.ones((height, width, 3), dtype=np.uint16) * 20000
    depth_map = np.ones((height, width, 3), dtype=np.uint16) * 20000
    res = np.zeros((height, width, 3), dtype=np.uint8)
    tem = np.zeros((height, width, 3), dtype=np.uint8)
    # for pc, face_index, t_u, t_v, x, y, z, a, b, c in zip(pcs, face_indexs, t_us, t_vs, xs, ys, zs, aas, bs, cs):
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
            for p, bbox in enumerate(part_bboxs):

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
            if is_render_part:
                res[min_v:max_v, min_u:max_u, 0:1][append_mask] = part_index+1
                res[min_v:max_v, min_u:max_u, 1:2][append_mask] = part_index+1
                res[min_v:max_v, min_u:max_u, 2:3][append_mask] = part_index+1
            else:
                res[min_v:max_v, min_u:max_u, 0:1][append_mask] = 1
                res[min_v:max_v, min_u:max_u, 1:2][append_mask] = 1
                res[min_v:max_v, min_u:max_u, 2:3][append_mask] = 1

    mask = ((res[:, :, 0] > 0) & (res[:, :, 1] > 0))

    return res, mask


def append_uv_map(uv_map, img, alpha=0.3):
    heat_map = np.zeros((img.shape[0], img.shape[1], 3))
    color_bar = cv2.imread('./recon/bar.png')[3:, :, :]
    uv = np.argwhere(uv_map > 0)
    # print(uv.shape)
    for k in range(uv.shape[0]):
        i = uv[k][0]
        j = uv[k][1]
        heat_map[i][j][0] = color_bar[int((1-uv_map[i][j])*color_bar.shape[0]-1)][10][0]
        heat_map[i][j][1] = color_bar[int((1-uv_map[i][j])*color_bar.shape[0]-1)][10][1]
        heat_map[i][j][2] = color_bar[int((1-uv_map[i][j])*color_bar.shape[0]-1)][10][2]
			# heat_map[i][j][3] = 255

    append_area = uv_map > 0
    img[append_area] = heat_map[append_area] * alpha + img[append_area] * (1-alpha)	

    return img