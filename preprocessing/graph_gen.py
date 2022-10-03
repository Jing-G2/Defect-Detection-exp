import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import multiprocessing as mp

datasets_root = "/nfs4-p1/gj/datasets/"
# dataset_name = "NEU-CLS"
graph_data_root = "/nfs4-p1/gj/DEFECT2022/data0/"

# class_names = ["RS", "In", "Pa", "Sc", "PS", "Cr"]
# image_size = 200  # hight = width
# class_num = 300  # 300 images in one class
# unit_size = 10
# color_unit_size = 16

dataset_name = "NEU-CLS-64"
class_names = os.listdir(os.path.join(datasets_root, dataset_name))
image_size = 64  # hight = width
unit_size = 8
color_unit_size = 16


def func(class_name, img_name):
    img = plt.imread(
        os.path.join(datasets_root, dataset_name, class_name, img_name))
    img = img // color_unit_size

    nodes = []
    patch_x_max = int(image_size / unit_size)
    patch_y_max = patch_x_max

    for i in range(patch_x_max):
        for j in range(patch_y_max):
            patch = img[i * unit_size:(i + 1) * unit_size,
                        j * unit_size:(j + 1) * unit_size]
            uni_c, counts = np.unique(patch, return_counts=True)

            for c, count in zip(uni_c, counts):
                cur_node = dict()
                cur_node['i'] = np.array(i)
                cur_node['j'] = np.array(j)
                cur_node['c'] = np.array(c)
                cur_node['density'] = count / patch.size
                # cur_node['density'] = count

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
        if len(adj_nodes.index) > 27:
            print(node)
            print(adj_nodes)
            print(len(adj_nodes.index))
        for adj_nodes_index in adj_nodes.index:
            if adj_nodes_index != node_index:
                cur_edge = dict()
                cur_edge['from_node'] = node_index
                cur_edge['to_node'] = adj_nodes_index

                edges.append(cur_edge)

    edgesframe = pd.DataFrame(edges)

    dst_path = os.path.join(graph_data_root, dataset_name, class_name)

    nodesframe.to_csv(os.path.join(dst_path, img_name + '_nodes.csv'),
                      index_label='node_id')
    edgesframe.to_csv(os.path.join(dst_path, img_name + '_edges.csv'),
                      index=False)
    del nodesframe
    del edgesframe


def img_convert(class_name, img_list):
    if not os.path.exists(
            os.path.join(graph_data_root, dataset_name, class_name)):
        os.makedirs(os.path.join(graph_data_root, dataset_name, class_name),
                    exist_ok=True)

    count = 0

    for img_name in img_list:
        func(class_name, img_name)
        count += 1

    print(count)


if __name__ == '__main__':
    num_cores = int(mp.cpu_count())
    print("#cores: ", num_cores)
    pool = mp.Pool(num_cores)
    print(graph_data_root)
    para = []

    for class_name in class_names:
        # for i in range(1, class_num + 1):
        # img_name = class_name + '_' + str(i)
        # img = plt.imread(
        #     os.path.join(datasets_root, dataset_name, img_name + '.bmp'))
        img_list = os.listdir(
            os.path.join(datasets_root, dataset_name, class_name))
        para.append((class_name, img_list))
        print(len(img_list))

    pool.starmap_async(img_convert, para )
