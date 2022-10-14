import glob
import os
import scipy.io as scio
import numpy as np
import cv2
import pandas as pd
import multiprocessing as mp
from math import ceil

datasets_root = "/nfs4-p1/gj/datasets/AnomalyDetection"
graph_data_root = "/nfs4-p1/gj/DEFECT2022/data/"
class_names = ['Normal', 'Anomalous']

dataset_name = "CrackForest"
unit_size = 8
color_unit_size = 16


def func(img, img_name, class_index):
    height, width = img.shape[0], img.shape[1]
    img = img // color_unit_size

    patch_x_max = ceil(height / unit_size)
    patch_y_max = ceil(width / unit_size)

    nodes = []
    # divide the whole image into several patch
    for i in range(patch_x_max):
        for j in range(patch_y_max):
            if (i + 1) * unit_size < height and (j + 1) * unit_size < width:
                patch = img[i * unit_size:(i + 1) * unit_size,
                            j * unit_size:(j + 1) * unit_size]
            elif (i + 1) * unit_size >= height and (j + 1) * unit_size < width:
                patch = img[i * unit_size:, j * unit_size:(j + 1) * unit_size]
            elif (i + 1) * unit_size < height and (j + 1) * unit_size >= width:
                patch = img[i * unit_size:(i + 1) * unit_size, j * unit_size:]
            elif (i + 1) * unit_size >= height and (j +
                                                    1) * unit_size >= width:
                patch = img[i * unit_size:, j * unit_size:]

            uni_c, counts = np.unique(patch, return_counts=True)

            # define each patch as a node in the graph
            for c, count in zip(uni_c, counts):
                density = count / patch.size
                if density < 0.05:
                    continue
                cur_node = dict()
                cur_node['i'] = np.array(i)
                cur_node['j'] = np.array(j)
                cur_node['c'] = np.array(c)
                cur_node['density'] = density * unit_size * unit_size

                nodes.append(cur_node)

    nodesframe = pd.DataFrame(nodes)

    edges = []
    for node_index in range(nodesframe.shape[0]):
        node = nodesframe.iloc[node_index]
        adj_nodes = nodesframe.loc[(nodesframe['i'] >= node['i'] - 1)
                                   & (nodesframe['i'] <= node['i'] + 1) &
                                   (nodesframe['j'] >= node['j'] - 1) &
                                   (nodesframe['j'] <= node['j'] + 1) &
                                   (nodesframe['c'] >= node['c'] - 1) &
                                   (nodesframe['c'] <= node['c'] + 1)]

        for adj_nodes_index in adj_nodes.index:
            if adj_nodes_index != node_index:
                cur_edge = dict()
                cur_edge['from_node'] = node_index
                cur_edge['to_node'] = adj_nodes_index

                edges.append(cur_edge)

    edgesframe = pd.DataFrame(edges)

    dst_path = os.path.join(graph_data_root, dataset_name,
                            class_names[class_index])
    nodesframe.to_csv(os.path.join(dst_path, img_name + '_nodes.csv'),
                      index_label='node_id')
    edgesframe.to_csv(os.path.join(dst_path, img_name + '_edges.csv'),
                      index=False)

    del nodes
    del edges
    del nodesframe
    del edgesframe


def img_convert(groundTruth, image):
    gt = scio.loadmat(groundTruth)['groundTruth'][0, 0][0]
    img = cv2.imread(image)
    img_name = image.split('/')[-1].split('.')[0]

    # split the whole image(320*480) into 6 patches (80*80)
    for i in range(320 // 80):
        for j in range(480 // 80):
            img_patch = img[i * 80:(i + 1) * 80, j * 80:(j + 1) * 80]
            gt_patch = gt[i * 80:(i + 1) * 80, j * 80:(j + 1) * 80]
            class_index = 1 if 2 in np.unique(gt_patch) else 0
            patch_name = img_name + '_{}_{}'.format(i, j)

            func(img_patch, patch_name, class_index)

            cv2.imwrite(
                os.path.join(datasets_root, dataset_name,
                             class_names[class_index], patch_name + '.jpg'),
                img_patch)
    print(img_name)


if __name__ == '__main__':
    num_cores = int(mp.cpu_count())
    print("#cores: ", num_cores)
    pool = mp.Pool(num_cores)
    print(graph_data_root)

    if not os.path.exists(
            os.path.join(datasets_root, dataset_name, class_names[0])):
        os.makedirs(os.path.join(datasets_root, dataset_name, class_names[0]))
        os.makedirs(os.path.join(datasets_root, dataset_name, class_names[1]))
    if not os.path.exists(
            os.path.join(graph_data_root, dataset_name, class_names[0])):
        os.makedirs(os.path.join(graph_data_root, dataset_name,
                                 class_names[0]))
        os.makedirs(os.path.join(graph_data_root, dataset_name,
                                 class_names[1]))

    groundTruth_list = glob.glob(
        os.path.join(datasets_root, dataset_name, 'groundTruth',
                     f'*.mat'))
    img_list = glob.glob(
        os.path.join(datasets_root, dataset_name, 'image', f'*.jpg'))

    para = [(groundTruth, image)
            for groundTruth, image in zip(groundTruth_list, img_list)]
    print(len(para))
    for gt, img in zip(groundTruth_list, img_list):
        img_convert(gt, img)

    # pool.starmap_async(img_convert, para)
    # pool.close()
    # pool.join()
