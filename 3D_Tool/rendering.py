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
        [113, 165, 0],
        [8, 78, 183],
        [112, 252, 57],
        [5, 28, 126],
        [100, 111, 156],
        [140, 60, 39],
        [75, 13, 159],
        [188, 110, 83]
    ]
    depth_map = np.ones((height, width)) * 50000
    res = np.zeros((height, width, 3), dtype=np.uint8)

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
            for p, bbox in enumerate(self.part_bboxs):
                if (min_tv + max_tv) / 2 > bbox[0] and (min_tu + max_tu) / 2 > bbox[1] and (min_tv + max_tv) / 2 < bbox[
                    2] and (min_tu + max_tu) / 2 < bbox[3]:
                    # if min_tv > bbox[0] and min_tu > bbox[1] and max_tv < bbox[2] and max_tu < bbox[3]:
                    part_index = p

            append_mask = des < depth_map[min_v:max_v, min_u:max_u]
            depth_map[min_v:max_v, min_u:max_u][append_mask] = des

            res[min_v:max_v, min_u:max_u, 0:1][append_mask] = color_table[part_index][2]
            res[min_v:max_v, min_u:max_u, 1:2][append_mask] = color_table[part_index][1]
            res[min_v:max_v, min_u:max_u, 2:3][append_mask] = color_table[part_index][0]

    mask = (depth_map < 50000)

    return res, mask