import numpy as np
import os
for name in os.listdir('F:\BaiduNetdiskDownload\kitti_model\simplification\\'):
    point_cloud_path = 'F:\BaiduNetdiskDownload\kitti_model\simplification\\' + name
    print(name)
    if name == 'Notchback_BD247_Buick_rega.obj':
        continue
    v = []
    vt = []
    vn = []
    face_v = []
    face_t = []

    with open(point_cloud_path) as f:
        line = f.readline()
        while line:
            if line[0] == 'v' and line[1] == ' ':
                v.append(
                    [float(item) for item in line.split()[1:]]
                )

            elif line[0] == 'f':
                item = line.split(' ')
                f1 = int(item[1].split('/')[0]) - 1
                f2 = int(item[2].split('/')[0]) - 1
                f3 = int(item[3].split('/')[0]) - 1
                face_v.append([f1, f2, f3])
                f1_t = int(item[1].split('/')[1]) - 1
                f2_t = int(item[2].split('/')[1]) - 1
                f3_t = int(item[3].split('/')[1]) - 1
                face_t.append([f1_t, f2_t, f3_t])

            elif line[0] == 'v' and line[1] == 't':
                vt.append(
                    [float(item) for item in line.split()[1:]]
                )
            elif line[0] == 'v' and line[1] == 'n':
                vn.append(
                    [float(item) for item in line.split()[1:]]
                )
            line = f.readline()

    new_vt = [[0, 0] for i in range(len(vt))]
    for f_v, f_t in zip(face_v, face_t):
        if len(f_v) != 3:
            print(len(f_v))
            print(f_v)
        for i in range(3):
            new_vt[f_v[i]] = vt[f_t[i]]

    with open('F:\BaiduNetdiskDownload\kitti_model\simplifacation_reorganize\\' + name, 'w') as f:
        for vv in v:
            f.write('v ' + str(vv[0]) + ' ' + str(vv[1]) + ' ' + str(vv[2]) + '\n')
        for vnn in vn:
            f.write('vn ' + str(vnn[0]) + ' ' + str(vnn[1]) + ' ' + str(vnn[2]) + '\n')
        for vtt in new_vt:
            f.write('vt ' + str(vtt[0]) + ' ' + str(vtt[1]) + '\n')
        for f_v in face_v:
            f.write('f ' + str(f_v[0] + 1) + '/' + str(f_v[0] + 1) + '/' + str(f_v[0] + 1) + ' '
                    + str(f_v[1] + 1) + '/' + str(f_v[1] + 1) + '/' + str(f_v[1] + 1) + ' '
                    + str(f_v[2] + 1) + '/' + str(f_v[2] + 1) + '/' + str(f_v[2] + 1) + '\n')
