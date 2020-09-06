import pickle
from densepose.structures import DensePoseResult
from recon.recon import estimate_6Dof_pose_with_partiou, get_camera_mat, papare_sim_models, get_color, papare_for_reconstruction, convert_part_uv_to_global_uv, flann_match_keypoints,get_pc_from_depth, get_texture_part_color, convert_part_uv_to_global_uv_sparse, estimate_6Dof_pose, papare_template_vertexs, estimate_6Dof_pose_multiprocess, get_camera_mat,vis_part, append_uv_map
import math
import cv2
import os
import time
from multiprocessing import Process
import numpy as np
from tqdm import tqdm
import argparse
import yaml
from skp_rpn.skp_3d_detnet import SKP3DDetNet
from skp_rpn.utils import get_kp_matrix, get_kp_uv, flann_match_structurekeypoints, convert_part_uv_to_global_uv_for_skp
import torch
from torchvision.transforms import functional as F


def load_yaml_cfg(cfg):
	with open(cfg,'r',encoding='utf-8') as f:
		cfg = f.read()
		d = yaml.load(cfg)
		return d

def vis_part_uv(img_part, img_u, img_v, part_img, u_map, v_map, bbox_xyxy):
    part_mask = np.zeros((img_part.shape[0], img_part.shape[1]))
    part_mask[int(bbox_xyxy[1]):int(bbox_xyxy[1]) + part_img.shape[0], int(bbox_xyxy[0]):int(bbox_xyxy[0]) + part_img.shape[1]] = part_img
    vis_part(part_mask, img_part)

    u_mask = np.zeros((img_part.shape[0], img_part.shape[1]))
    u_mask[int(bbox_xyxy[1]):int(bbox_xyxy[1]) + part_img.shape[0], int(bbox_xyxy[0]):int(bbox_xyxy[0]) + part_img.shape[1]] = u_map
    append_uv_map(u_mask, img_u)

    v_mask = np.zeros((img_part.shape[0], img_part.shape[1]))
    v_mask[int(bbox_xyxy[1]):int(bbox_xyxy[1]) + part_img.shape[0], int(bbox_xyxy[0]):int(bbox_xyxy[0]) + part_img.shape[1]] = v_map
    append_uv_map(v_mask, img_v)

def save_models(best_shape_pc, others_str, instance_id, img_name, save_dir):
    '''
    best_shape_pc (numpy) shape = [3,n]
    others_str [str]
    '''
    with open(os.path.join(save_dir, img_name.split('.')[0] +'_' + str(instance_id) + '.obj'), 'w') as f:
        for i in range(best_shape_pc.shape[1]):
            f.write('v ' + str(best_shape_pc[0][i]) + ' ' + str(best_shape_pc[1][i]) + ' ' + str(best_shape_pc[2][i]) + '\n')
        for other in others_str:
            f.write(other)

def mk_res_dir(cfg):
    if not os.path.exists(cfg['part_seg_output_dir']):
        os.makedirs(cfg['part_seg_output_dir'])
    if not os.path.exists(cfg['uv_reg_output_dir']):
        os.makedirs(cfg['uv_reg_output_dir'])
    if not os.path.exists(cfg['re_render_ouput_dir']):
        os.makedirs(cfg['re_render_ouput_dir'])
    if not os.path.exists(cfg['pos_res_output_dir']):
        os.makedirs(cfg['pos_res_output_dir'])
    if not os.path.exists(cfg['rencon_output_dir']):
        os.makedirs(cfg['rencon_output_dir'])

def get_pose_from_skpnet(skp_model, img_name, uv, uv_in_raw, kp_uv, depth_img):
    us, vs = flann_match_structurekeypoints(uv, kp_uv, uv_in_raw)
    kp_mat = get_kp_matrix(us, vs, depth_img)

    kp_mat = kp_mat.astype('float32')
    kp_mat = F.to_tensor(kp_mat).to('cuda')
    kp_mat = torch.unsqueeze(kp_mat, 0)
    _, scores, positions, sizes = skp_model(kp_mat)
    positions = torch.squeeze(positions[0].cpu())
    score = torch.squeeze(scores[0].cpu())
    sizes = torch.squeeze(sizes[0].cpu())
    x, y, z, b = float(positions[0]), float(positions[1]), float(positions[2]), float(positions[3])
    height, width, length = float(sizes[0]), float(sizes[1]), float(sizes[2])
    score =  1.0/(math.exp(-float(score)) + 1.0)
    return x, y, z, height, width, length, b, score

def sovle_pose_and_shape(data, skp_model, pcs, car_names, part_bboxes, spcs, face_indexs, t_us, t_vs, kp_uv, others_str, cfg):
    for img_id in tqdm(range(0, len(data))):
        target_num = len(data[img_id]['pred_boxes_XYXY'])
        start = time.time()
        img_name = os.path.basename(data[img_id]['file_name'])
        depth_img = cv2.imread(os.path.join(cfg['est_depth_dir'], img_name))
        img = cv2.imread(os.path.join(cfg['input_image_dir'], img_name))
        if depth_img is None or img is None:
            continue
        depth_img = cv2.resize(depth_img, (img.shape[1], img.shape[0]))
        depth_img = depth_img[:, :, 0]
        
        img_part = np.copy(img)
        img_u = np.copy(img)
        img_v = np.copy(img)

        for instance_id in range(target_num):
            bbox_xyxy = data[img_id]['pred_boxes_XYXY'][instance_id]
            # pre_class = data[img_id]['pred_classes'][instance_id]
            result_encoded = data[img_id]['pred_densepose'].results[instance_id]
            iuv_arr = DensePoseResult.decode_png_data(*result_encoded)

            # part segmentation, shape = [bbox_hegiht, bbox_width]
            part_img = iuv_arr[0,:,:]
            part_mask = np.zeros((img.shape[0], img.shape[1]))
            part_mask[int(bbox_xyxy[1]):int(bbox_xyxy[1]) + part_img.shape[0], int(bbox_xyxy[0]):int(bbox_xyxy[0]) + part_img.shape[1]] = part_img
            # u_map rescale to [0, 1], shape = [bbox_hegiht, bbox_width]
            u_map = iuv_arr[1,:,:] / 255.0
            # v_map rescale to [0, 1], shape = [bbox_hegiht, bbox_width]
            v_map = iuv_arr[2,:,:] / 255.0

            vis_part_uv(img_part, img_u, img_v, part_img, u_map, v_map, bbox_xyxy)
            
            # for each pix in car instance, convert it to uv(coordinate in texute_map) and uv_in_raw(coordinate in input image)
            uv, uv_in_raw = convert_part_uv_to_global_uv(u_map, v_map, part_img, part_bboxes)
            
            uv_in_raw[:, 0] += int(bbox_xyxy[0])
            uv_in_raw[:, 1] += int(bbox_xyxy[1])

            # for each pix in car instance, find its corresponding 3D vertex index in template
            sample_count = min(len(uv), 4096)
            indexs = np.arange(len(uv))
            np.random.shuffle(indexs)
            new_uv = []
            new_uv_in_raw = []
            for s in indexs[0:sample_count]:
                new_uv_in_raw.append(uv_in_raw[s])
                new_uv.append(uv[s])
            vertexs_index = flann_match_keypoints(new_uv, target_uv, texture)


            # instance is too small to caculate, so drop it
            if len(new_uv_in_raw) < 10:
                continue
            camera_mat = get_camera_mat(img_name, cfg['calib_dir'])

            img, detection_res, best_shape = estimate_6Dof_pose_with_partiou(vertexs_index, np.array(new_uv_in_raw), img, pcs, car_names, bbox_xyxy, img_name, spcs, face_indexs, t_us, t_vs, part_mask, part_bboxes, camera_mat)
            [bbox1, bbox2, bbox3, bbox4, height, width, length, x, y, z, b, s] = detection_res
            save_models(best_shape, others_str, instance_id, img_name, cfg['rencon_output_dir'])

            uv, uv_in_raw = convert_part_uv_to_global_uv_for_skp(u_map, v_map, part_img, part_bboxes, bbox_xyxy)
            x, y, z, height, width, length, b, s = get_pose_from_skpnet(skp_model, img_name, uv, uv_in_raw, kp_uv, depth_img)

            with open(os.path.join(cfg['pos_res_output_dir'], img_name.split('.')[0] + '.txt'), 'a') as f:
                    f.write('Car -1 -1 -10 ' + str(float(bbox1)) + ' ' + str(float(bbox2)) + ' ' + str(float(bbox3)) + ' ' + str(float(bbox4)) + ' ' + 
                    str(height) + ' ' + str(width) + ' ' + str(length) + ' ' + 
                    str(x) + ' ' + str(y) + ' ' + str(z) + ' ' + str(b) + ' ' + str(float(s)) + '\n')
            
        cv2.imwrite(os.path.join(cfg['re_render_ouput_dir'], img_name), img)
        cv2.imwrite(os.path.join(cfg['part_seg_output_dir'], img_name.split('.')[0] + '_part.png'), img_part)
        cv2.imwrite(os.path.join(cfg['uv_reg_output_dir'], img_name.split('.')[0] + '_u.png'), img_u)
        cv2.imwrite(os.path.join(cfg['uv_reg_output_dir'], img_name.split('.')[0] + '_v.png'), img_v)

        
parser = argparse.ArgumentParser(description="pose solver")
parser.add_argument('--cfg', default='config.yaml', help='config_file')
args = parser.parse_args()
cfg = load_yaml_cfg(args.cfg)
# part_bboxes: pre-defined part bbox in texture map
# target_uv: croodinate of each template 3d vertex in texture map 
part_bboxes, target_uv, model_face_uv_str = papare_for_reconstruction(cfg['temaplate_models_dir'], cfg['texture_path'])
pcs, car_names = papare_template_vertexs(cfg['temaplate_models_dir'])
spcs, face_indexs, t_us, t_vs = papare_sim_models(cfg['simplification_temaplate_models_dir'])
test_dir = cfg['input_image_dir']
texture = cv2.imread(cfg['texture_path'])

mk_res_dir(cfg)
# load densepose network inferring result
f = open(cfg['stage1_network_res'], 'rb')
data = pickle.load(f)

skp_model = SKP3DDetNet()
skp_model.load_state_dict(torch.load(cfg['skp_model_path'])['model'])
skp_model.to('cuda')
skp_model.eval()
kp_uv = get_kp_uv(target_uv)
sovle_pose_and_shape(data, skp_model, pcs, car_names, part_bboxes, spcs, face_indexs, t_us, t_vs, kp_uv, model_face_uv_str, cfg)
