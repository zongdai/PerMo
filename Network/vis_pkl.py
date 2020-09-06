import pickle
from densepose.structures import DensePoseResult
from recon.recon import vis_part, append_uv_map

import cv2
import os
import argparse
import yaml
import numpy as np
from tqdm import tqdm

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

def mk_res_dir(cfg):
    if not os.path.exists(cfg['part_seg_output_dir']):
        os.makedirs(cfg['part_seg_output_dir'])
    if not os.path.exists(cfg['uv_reg_output_dir']):
        os.makedirs(cfg['uv_reg_output_dir'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vis pkl")
    parser.add_argument('--cfg', default='config.yaml', help='config_file')
    args = parser.parse_args()
    cfg = load_yaml_cfg(args.cfg)

    mk_res_dir(cfg)
    # load densepose network inferring result
    f = open(cfg['stage1_network_res'], 'rb')
    data = pickle.load(f)
    test_dir = cfg['input_image_dir']

    for img_id in tqdm(range(0, len(data))):
        target_num = len(data[img_id]['pred_boxes_XYXY'])
        img_name = os.path.basename(data[img_id]['file_name'])
    
        img = cv2.imread(test_dir + img_name)
        img_part = np.copy(img)
        img_u = np.copy(img)
        img_v = np.copy(img)
        # camera_mat = get_camera_mat(img_name)
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

        cv2.imwrite(os.path.join(cfg['part_seg_output_dir'], img_name.split('.')[0] + '_part.png'), img_part)
        cv2.imwrite(os.path.join(cfg['uv_reg_output_dir'], img_name.split('.')[0] + '_u.png'), img_u)
        cv2.imwrite(os.path.join(cfg['uv_reg_output_dir'], img_name.split('.')[0] + '_v.png'), img_v)
