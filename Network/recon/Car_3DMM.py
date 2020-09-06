# import scipy.io as scio
import numpy as np
import math
import random
import cv2
import os

# read a obj file with scale
#
# return 4*n shape martix
def read_vertexs(file, scale):
    vertexs = []
    with open(file) as f:
        line = f.readline()
        while line:
            if line[0] == 'v':
                items = line.split(' ')
                vertex = [float(items[1])*scale,float(items[2])*scale, float(items[3].split('\n')[0])*scale, 1.0]

                vertexs.append(vertex)
            line = f.readline()
    vertexs_np = np.transpose(np.array(vertexs))
    # print(vertexs_np.shape)
    return vertexs_np

def rescale_pca_shape_coeff(pca_shape_coeff, pca_value):
    for i in range(0, pca_shape_coeff.shape[0]):
        for j in range(0, pca_shape_coeff.shape[1]):
            pca_shape_coeff[i][j] = pca_shape_coeff[i][j] / math.sqrt(pca_value[j][0])

# read obj face index
# like :
# f 1 2 3
# f 2 3 4
def read_face_index(model_path):
    with open(model_path) as f:
        line = f.readline()
        face_index = []
        while(line):
            if line[0] == 'f':
                items = line.split(' ')
                face_index.append([(items[1]), (items[2]), (items[3].split('\n')[0])])
            line = f.readline()
    return face_index

# get 3d bounding box
# vertexs is np and its shape is 3n*1
# return 8 points
# line 0-1, 0-3, 1-2, 2-3, 4-5, 5-6, 6-7, 7-4, 0-4, 1-5, 2-6, 3-7 to generate a box
def get_model_bounding_box(vertexs):
    x_np = vertexs[0, :]
    y_np = vertexs[1, :]
    z_np = vertexs[2, :]

    max_x = np.max(x_np)
    min_x = np.min(x_np)
    max_y = np.max(y_np)
    min_y = np.min(y_np)
    max_z = np.max(z_np)
    min_z = np.min(z_np)

    p1 = [max_x, max_y, max_z, 1]
    p2 = [max_x, min_y, max_z, 1]
    p3 = [min_x, min_y, max_z, 1]
    p4 = [min_x, max_y, max_z, 1]
    p5 = [max_x, max_y, min_z, 1]
    p6 = [max_x, min_y, min_z, 1]
    p7 = [min_x, min_y, min_z, 1]
    p8 = [min_x, max_y, min_z, 1]

    bouding = np.array([p1,p2,p3,p4,p5,p6,p7,p8])
    return bouding.T


def write_pca_model(write_path, vertexs, face_index, other_text = None):
    vertexs = vertexs[:, 0]
    with open(write_path, 'w') as f:
        for i in range(0, len(vertexs)//3):
            f.write("v " + str(vertexs[i*3]) + " " + str(vertexs[i*3+1]) + " " + str(vertexs[i*3+2]) + "\n")
        if other_text is not None:
            for t in other_text:
                f.write(t)
        else:
            for face in face_index:
                f.write("f " + face[0] + " " + face[1] + " " + face[2] + "\n")

def write_pca_model_with_rt(write_path, vertexs, face_index, rt, other_text = None):
    vertexs = vertexs[:, 0]
    v = []
    for i in range(0, len(vertexs) // 3):
        v.append([vertexs[i*3], vertexs[i*3+1], vertexs[i*3+2], 1])
    np_v = (np.array(v)).T
    new_v = np.dot(rt, np_v)
    with open(write_path, 'w') as f:
        for i in range(0, new_v.shape[1]):
            f.write("v " + str(new_v[0][i]) + " " + str(new_v[1][i]) + " " + str(new_v[2][i]) + "\n")
        if other_text is not None:
            for t in other_text:
                f.write(t)
        else:
            for face in face_index:
                f.write("f " + face[0] + " " + face[1] + " " + face[2] + "\n")

# generate a model with mean, eigen_vectors, pca_coeff
# return a vertex set with 3n*1
def generate_pca_model(pca_mean_value, pca_eigen_vectors, pca_shape_coeff):

    p = np.dot(pca_eigen_vectors, pca_shape_coeff)
    p = np.resize(p, (p.shape[0], 1))
    # print(p.shape)
    new_model = pca_mean_value + p
    return new_model

# random combin 2 coeffs from 34 coeffs
def generate_pca_coff(pca_shape_coeff):
    new_pca_shape_coeff = (pca_shape_coeff[random.randint(0, len(pca_shape_coeff)-1)] + pca_shape_coeff[random.randint(0, len(pca_shape_coeff)-1)])/2
    return new_pca_shape_coeff

def generate_pca_coff_single(pca_shape_coeff, index, iteration):
    min_value = np.min(pca_shape_coeff[:,2])
    max_value = np.max(pca_shape_coeff[:,2])
    new_pca_shape_coeff = np.zeros((19, 1))
    new_pca_shape_coeff[2][0] = min_value + (max_value - min_value)/iteration * index * 2
    return new_pca_shape_coeff

# read u-v to vertex index mapper file
# read [u, v] and [index]
def read_mapper_file(mapper_file_path):
    point_2d = []
    vertex_indexs = []
    with open(mapper_file_path) as f:
        line = f.readline()
        while line:
            items = line.split(' ')
            point_2d.append([int(items[2]), int(items[1])])
            vertex_index = int(items[3].split('\n')[0])
            vertex_indexs.append(vertex_index)
            line = f.readline()

    return np.float32(point_2d), vertex_indexs

def R_Mat_to_Euler(R):
    # float sy = sqrt(R.at<double>(0,0) * R.at<double>(0,0) +  R.at<double>(1,0) * R.at<double>(1,0) );
    sy = math.sqrt(R[0][0] * R[0][0] + R[1][0] * R[1][0])
    if sy > 1e-6:
        x = math.atan2(R[2][1], R[2][2])
        y = math.atan2(-R[2][0], sy)
        z = math.atan2(R[1][0], R[0][0])
    else:
        x = math.atan2(-R[1][2], R[1][1])
        y = math.atan2(-R[2][0], sy)
        z = 0
    
    return x, y, z


# get projection_matrix, Rt and pose
def get_projection_matrix(camera_matrix, point_2d, point_3d, reprojectionError):
    # print(point_2d.shape)
    # print(point_3d.shape)
    point_2d = np.float32(point_2d)
    retval, rvec, tvec, inliers = cv2.solvePnPRansac(point_3d, point_2d, camera_matrix, np.zeros((5,1)), iterationsCount=1000, reprojectionError=reprojectionError, flags=1)
    # print(len(inliers))
    # _,rvec, tvec = cv2.solvePnP(point_3d, point_2d, camera_matrix, np.zeros((5, 1)))
    pose = [rvec[0][0], rvec[1][0], rvec[2][0] , tvec[0][0], tvec[1][0], tvec[2][0]]
    rvec, jac = cv2.Rodrigues(rvec)
    a, b, c = R_Mat_to_Euler(rvec)
    pose[0] = a
    pose[1] = b
    pose[2] = c
    if pose[5] < 0:
        pose[0] += math.pi
        pose[3] = -pose[3]
        pose[4] = -pose[4]
        pose[5] = -pose[5]

    Rt = np.hstack((rvec, tvec))
    return np.dot(camera_matrix, Rt), Rt, pose

# get 3d point with indexs
def get_point3d(vertex_indexs, model_vertex):
    point_3d = []
    for vertex_index in vertex_indexs:
        point_3d.append([model_vertex[vertex_index * 3][0], model_vertex[vertex_index * 3 + 1][0],
                         model_vertex[vertex_index * 3 + 2][0]])
    return np.float32(point_3d)

def get_point3d2(vertex_indexs, model_vertex):
    # print(model_vertex.shape)
    point_3d = []
    for vertex_index in vertex_indexs:
        point_3d.append([model_vertex[0][vertex_index], model_vertex[1][vertex_index],
                         model_vertex[2][vertex_index]])
    return np.float32(point_3d)

# caculate projection error
def caculate_error(point_3d, tar_u, tar_v, projection_matrix):
    # print(point_3d.shape)
    E = np.ones((1, point_3d.shape[1]))
    new_point_3d = np.vstack((point_3d, E))
    # print(new_point_3d.shape)
    p = np.dot(projection_matrix, new_point_3d)
    u = p[0,:] / p[2,:]
    v = p[1,:] / p[2,:]
    error = np.linalg.norm(u - tar_u) + np.linalg.norm(v - tar_v)
    return error

# generate some coeff from 34 template
def random_coeff(pca_shape_coeff, random_value_num = 2,random_row_count = 5):

    coeff = []
    min_value_0 = np.min(pca_shape_coeff[:, 0])
    max_value_0 = np.max(pca_shape_coeff[:, 0])
    min_value_1 = np.min(pca_shape_coeff[:, 1])
    max_value_1 = np.max(pca_shape_coeff[:, 1])

    for i in range(0,random_row_count+1):
        for j in range(0, random_row_count+1):
            # for k in range(0, random_row_count + 1):
                new_pca_shape_coeff = np.zeros((19, 1))
                new_pca_shape_coeff[0][0] = min_value_0 + (max_value_0 - min_value_0) / random_row_count * i
                new_pca_shape_coeff[1][0] = min_value_1 + (max_value_1 - min_value_1) / random_row_count * j
                coeff.append(new_pca_shape_coeff)

    return coeff

# backprojection to img
def projection2img(img_path, obj_path, projection_matrix, save_name, pix2point_dir):
    img_out = np.zeros((1080, 1920, 4))
    # pix2point_dir = "USA-Data\\connemara_way\\"
    new_point_3d = read_vertexs(obj_path, 1)
    p = np.dot(projection_matrix, new_point_3d)
    u = p[0, :] / p[2, :]
    v = p[1, :] / p[2, :]
    img = cv2.imread(img_path)
    for i, j in zip(u, v):
        # print(i,j)
        if abs(i) >= 1920 or abs(j) >= 1080:
            continue
        cv2.circle(img, (int(i), int(j)), 1, (0, 0, 255, 0.2), -1)
        cv2.circle(img_out, (int(i), int(j)), 1, (0, 0, 255), -1)
        img_out[int(j)][int(i)][3] = 180
    cv2.imwrite(pix2point_dir + "result\\" + save_name , img)
    cv2.imwrite(pix2point_dir + "result\\mask\\" + save_name , img_out)

def project_bounding2img(img, bounding,projection_matrix):
    p = np.dot(projection_matrix, bounding)
    v = p[1, :] / p[2, :]
    u = p[0, :] / p[2, :]
    v[v < 0] = 0
    v[v > img.shape[0]] = img.shape[0]
    u[u < 0] = 0
    u[u > img.shape[1]] = img.shape[1]
    min_u = np.min(u)
    min_v = np.min(v)
    color = (255, 0, 0)
    cv2.line(img, (int(u[0]), int(v[0])), (int(u[1]), int(v[1])), color, 1)
    cv2.line(img, (int(u[0]), int(v[0])), (int(u[3]), int(v[3])), color, 1)
    cv2.line(img, (int(u[1]), int(v[1])), (int(u[2]), int(v[2])), color, 1)
    cv2.line(img, (int(u[2]), int(v[2])), (int(u[3]), int(v[3])), color, 1)

    cv2.line(img, (int(u[4]), int(v[4])), (int(u[5]), int(v[5])), color, 1)
    cv2.line(img, (int(u[5]), int(v[5])), (int(u[6]), int(v[6])), color, 1)
    cv2.line(img, (int(u[6]), int(v[6])), (int(u[7]), int(v[7])), color, 1)
    cv2.line(img, (int(u[7]), int(v[7])), (int(u[4]), int(v[4])), color, 1)

    cv2.line(img, (int(u[0]), int(v[0])), (int(u[4]), int(v[4])), color, 1)
    cv2.line(img, (int(u[1]), int(v[1])), (int(u[5]), int(v[5])), color, 1)
    cv2.line(img, (int(u[2]), int(v[2])), (int(u[6]), int(v[6])), color, 1)
    cv2.line(img, (int(u[3]), int(v[3])), (int(u[7]), int(v[7])), color, 1)

    return img

def esitimate_pose_and_shape(pca_mean_value, pca_eigen_vectors, pca_shape_coeff, mapper_file_path, camera_matrix):
    point_2d, vertex_indexs = read_mapper_file(mapper_file_path)
    coeffs = random_coeff(pca_shape_coeff)
    min_error = 9e20
    best_coeff = []
    best_projection = []
    coeff_index = 0
    for coeff in coeffs:
        model_vertex = generate_pca_model(pca_mean_value.T, pca_eigen_vectors, coeff)
        point_3d = get_point3d(vertex_indexs, model_vertex)
        projection_matrix, rt , pose = get_projection_matrix(camera_matrix, point_2d, point_3d)
        error = caculate_error(point_3d, point_2d, projection_matrix)
        if error < min_error:
            # print(error)
            min_error = error
            best_coeff = coeff
            best_projection = projection_matrix
            best_rt = rt
            best_pose = pose
            best_coeff_index = coeff_index
        coeff_index += 1
    best_shape = generate_pca_model(pca_mean_value.T, pca_eigen_vectors, best_coeff)
    return best_shape, best_projection, best_rt, best_pose, best_coeff_index

def read_vertex2vector(model_path):
    vertexs = []
    with open(model_path) as f:
        line = f.readline()
        while line:
            if line[0] == 'v':
                items = line.split(' ')
                vertexs.append(float(items[1]))
                vertexs.append(float(items[2]))
                vertexs.append(float(items[3].split('\n')[0]))
            line = f.readline()
    vertexs_ = np.array(vertexs)

    return np.resize(vertexs_, (vertexs_.shape[0], 1))


def esitimate_pose_from_2models(model1, model2, mapper_file_path, camera_matrix):
    point_2d, vertex_indexs = read_mapper_file(mapper_file_path)
    point_3d1 = get_point3d(vertex_indexs, model1)
    projection_matrix1, rt1, pose1 = get_projection_matrix(camera_matrix, point_2d, point_3d1)
    error1 = caculate_error(point_3d1, point_2d, projection_matrix1)

    point_3d2 = get_point3d(vertex_indexs, model2)
    projection_matrix2, rt2, pose2 = get_projection_matrix(camera_matrix, point_2d, point_3d2)
    error2 = caculate_error(point_3d2, point_2d, projection_matrix2)
    print(error1, error2)
    result_pose = []
    if error1 > error2:
        result_pose = pose2
        result_projection = projection_matrix2
        result_rt  = rt2
    else:
        result_pose = pose1
        result_projection = projection_matrix1
        result_rt = rt1

    return result_pose, result_projection, result_rt

def get_TestDensepose(file_dir):
    imageFileNames = []
    pixel2pointFileNames = []
    for root, dirs, files in os.walk(file_dir):
        for f in files:
            if len(f.split(".txt")) == 2:
                pixel2pointFileNames.append(f)
    print(imageFileNames)
    print(pixel2pointFileNames)
    return imageFileNames, pixel2pointFileNames

def re_generate(imageFileNames, pixel2pointFileNames):
    with open("PoseResult.txt") as f:
        line = f.readline()
        i = 0
        while(line):
            items = pixel2pointFileNames[i].split('_')
            txtname = items[0] + "_" + items[1] + "_" + items[2] + "_" + items[3] + ".txt"
            with open("./pose_results/" + txtname, 'a') as f2:
            # with open("180116_054956530_pose.txt", 'a') as f2:

                instance_id =(pixel2pointFileNames[i].split('.')[0]).split('_')[4]
                f2.write(instance_id + " " + line)
            i += 1
            line = f.readline()

def write_pose(pose, pixel2point_name):
    items = pixel2point_name.split('_')
    file_name = items[0] + "_" + items[1] + "_" + items[2] + "_" + items[3] + ".txt"
    with open("poseresult\\" + file_name, "a") as f:
        index = pixel2point_name.split('_')[4].split('.')[0]
        f.write(index + " " + str(pose[0]) + " " + str(pose[1]) + " " + str(pose[2]) + " " + str(pose[3])  + " " + str(pose[4])  + " " + str(pose[5]) + "\n")


def show_mapper(img_path ,mapper_file_path):
    point_2d, vertex_indexs = read_mapper_file(mapper_file_path)
    img = cv2.imread(img_path)
    # print(point_2d.shape)
    for i in range(point_2d.shape[0]):
        # print(point_2d[i][0])
        cv2.circle(img, (int(point_2d[i][1]), int(point_2d[i][0])), 1, (0, 0, 255), -1)
    # cv2.imshow("img", img)
    # cv2.waitKey(0)

    return img
