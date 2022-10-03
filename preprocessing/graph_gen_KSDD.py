import glob
import os
from threading import Thread
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import multiprocessing as mp
from math import ceil

datasets_root = "/nfs4-p1/gj/datasets/AnomalyDetection/"
graph_data_root = "/nfs4-p1/gj/DEFECT2022/data0/"
class_names = ['Normal', 'Anomalous']

dataset_name = "KSDD"
dir_names = os.listdir(os.path.join(datasets_root, dataset_name))
unit_size = 20
color_unit_size = 16


def func(dir_name, img_index, img_path, label_path):
    img = plt.imread(img_path)
    label = plt.imread(label_path)
    class_index = 0 if label.max() == label.min() else 1

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
                cur_node = dict()
                cur_node['i'] = np.array(i)
                cur_node['j'] = np.array(j)
                cur_node['c'] = np.array(c)
                cur_node[
                    'density'] = count * unit_size * unit_size / patch.size

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
    nodesframe.to_csv(os.path.join(
        dst_path, dir_name + '_' + str(img_index) + '_nodes.csv'),
                      index_label='node_id')
    edgesframe.to_csv(os.path.join(
        dst_path, dir_name + '_' + str(img_index) + '_edges.csv'),
                      index=False)

    del nodesframe
    del edgesframe


def img_convert(dir_name, img_list, label_list):
    thread_list = []
    print(0, dir_name)
    for img_index in range(len(img_list)):
        func(dir_name, img_index, img_list[img_index], label_list[img_index])
    #     t = Thread(target=func,
    #                args=(dir_name, img_index, img_list[img_index],
    #                      label_list[img_index]))
    #     t.start()
    #     thread_list.append(t)

    # for t in thread_list:
    #     t.join()
    print(1, dir_name)


if __name__ == '__main__':
    num_cores = int(mp.cpu_count())
    print("#cores: ", num_cores)
    pool = mp.Pool(num_cores)
    print(graph_data_root)
    if not os.path.exists(
            os.path.join(graph_data_root, dataset_name, class_names[0])):
        os.makedirs(os.path.join(graph_data_root, dataset_name,
                                 class_names[0]))
        os.makedirs(os.path.join(graph_data_root, dataset_name,
                                 class_names[1]))
    para = []

    for dir_name in dir_names:
        label_list = glob.glob(
            os.path.join(datasets_root, dataset_name, dir_name, f'*.bmp'))
        img_list = glob.glob(
            os.path.join(datasets_root, dataset_name, dir_name, f'*.jpg'))
        # img_convert(dir_name, img_list, label_list)
        para.append((dir_name, img_list, label_list))
    pool.starmap_async(img_convert, para)
    pool.close()
    pool.join()
