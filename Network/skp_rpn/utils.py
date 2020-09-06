import cv2
import numpy as np

sample_count = [32, 32, 32, 32, 32, 32, 32, 0, 32, 32, 32, 32, 16, 16, 16, 16, 8, 8]


def flann_match_structurekeypoints(part_uv, kp_uv, uv_in_raw, img=None):
    match_vertexs_index = []
    kp_us = []
    kp_vs = []  
    # print(part_uv) 
    for part_index in range(18):
        if part_index in part_uv:
            # print(part_index)
    # for part_index, uv in part_uv.items():
            index_params = dict(algorithm=0, trees = 5)
            search_params = dict(checks=30)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
                
            matches = flann.knnMatch(np.float32(kp_uv[part_index]), np.float32(part_uv[part_index]), k=1)

            for _, m in enumerate(matches):
                if m[0].distance < 200:
                    uv_index = (m[0].trainIdx)
                    # print(uv_index)
                    # print(uv_in_raw[part_index].shape)
                    kp_us.append(uv_in_raw[part_index][uv_index][0])
                    kp_vs.append(uv_in_raw[part_index][uv_index][1])
                    
                else:
                    # print(m[0].distance)
                    kp_us.append(0)
                    kp_vs.append(0)
        else:
            for _ in range(sample_count[part_index]):
                kp_us.append(0)
                kp_vs.append(0)
    return kp_us, kp_vs


def get_kp_matrix(us, vs, depth_img):
    kp_mat = np.zeros((len(us), len(us), 3), dtype=np.int16)
    for i in range(len(us)):
        if us[i] == 0:
            continue
        for j in range(len(vs)):
            if vs[j] == 0:
                continue
            if i==j:
                kp_mat[i][j][0] = us[i]
                kp_mat[i][j][1] = vs[i]
                kp_mat[i][j][2] = depth_img[vs[i]][us[i]]
            else:
                kp_mat[i][j][0] = us[i] - us[j]
                kp_mat[i][j][1] = vs[i] - vs[j]
                # kp_mat[i][j][2] = int(math.sqrt((us[i] - us[j])*(us[i] - us[j]) + (vs[i] - vs[j])*(vs[i] - vs[j])))
                kp_mat[i][j][2] = (depth_img[vs[i]][us[i]] + depth_img[vs[j]][us[j]]) // 2
    return kp_mat

def get_kp_uv(model_uv):
    kp_uv = []
    with open('./skp_rpn/sample_index.txt') as f:
        indexs = [int(index) for index in f.readline().split()]
        indexx = 0
        for one_part_count in sample_count:
            kp_one_part_uv = []
            for i in range(one_part_count):
                kp_one_part_uv.append([model_uv[indexs[indexx]][0], model_uv[indexs[indexx]][1]])
                indexx += 1
            kp_uv.append(kp_one_part_uv)
    
    return kp_uv

def convert_part_uv_to_global_uv_for_skp(u_map, v_map, part_mask_img, bboxes, instance_bbox):
   
    parts_uv = {}
    uv_in_raw = {}
    
    for part_index in range(19):
        if part_index == 8 or part_index == 0:
            continue
        
        part_uv_in_raw = np.argwhere((part_mask_img == part_index))
        # part_mask = part_mask_img == part_index
        if part_uv_in_raw.shape[0] < 10:
            continue
        u = u_map[part_uv_in_raw[:,0], part_uv_in_raw[:,1]].reshape(-1, 1)
        v = v_map[part_uv_in_raw[:,0], part_uv_in_raw[:,1]].reshape(-1, 1)
        
        bbox = bboxes[part_index-1]
        u = bbox[1] + u*(bbox[3] - bbox[1])
        v = bbox[0] + v*(bbox[2] - bbox[0])
        uv = np.hstack((u, v))
        parts_uv[part_index-1] = uv
        part_uv_in_raw[:, [0,1]] = part_uv_in_raw[:, [1,0]]
        part_uv_in_raw[:, 0] += int(instance_bbox[0])
        part_uv_in_raw[:, 1] += int(instance_bbox[1])
        # print(part_uv_in_raw.shape)
        uv_in_raw[part_index-1] = part_uv_in_raw
    return parts_uv, uv_in_raw