import glob
import os
import scipy.io as scio
import numpy as np
import cv2
import pandas as pd
import multiprocessing as mp
import warnings

warnings.filterwarnings("ignore")
from math import ceil
# from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000

datasets_root = "/nfs4-p1/gj/datasets/AnomalyDetection"
graph_data_root = "/nfs4-p1/gj/DEFECT2022/data0/"
class_names = ['Normal', 'Anomalous']

dataset_name = "CrackForest"
unit_size = 8
color_unit_size = 16


def color_difference(color1, color2):
    rgb_color1 = sRGBColor(rgb_r=color1[0],
                           rgb_g=color1[1],
                           rgb_b=color1[2],
                           is_upscaled=True)
    rgb_color2 = sRGBColor(rgb_r=color2[0],
                           rgb_g=color2[1],
                           rgb_b=color2[2],
                           is_upscaled=True)
    lab_color1 = convert_color(rgb_color1, LabColor)
    lab_color2 = convert_color(rgb_color2, LabColor)
    delta_e = delta_e_cie2000(lab_color1, lab_color2)
    # fix no asscalar problem by replacing by ndarray.item
    return delta_e


def func(img, img_name, class_index):
    height, width = img.shape[0], img.shape[1]

    colors, counts = np.unique(img.reshape(-1, 3), return_counts=True, axis=0)
    distances = pairwise_distances(colors, metric=color_difference)
    tsne = TSNE(n_components=1, metric='precomputed',learning_rate='auto', n_iter=500)
    tsne_result = tsne.fit_transform(distances)
    min_tsne = tsne_result.min()
    max_tsne = tsne_result.max()

    # km = KMeans(n_clusters=16)
    # cluster  = km.fit_predict(tsne_result)
    # ----------------------------------------------------------
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

            if len(patch.shape) > 2:  # rgb
                uni_c, counts = np.unique(patch.reshape(-1, 3),
                                          return_counts=True,
                                          axis=0)
            else:  # gray
                uni_c, counts = np.unique(patch, return_counts=True)

            # define each patch as a node in the graph
            for c, count in zip(uni_c, counts):
                if count / (patch.shape[0] * patch.shape[1]) < 0.05:
                    continue
                cur_node = dict()
                cur_node['i'] = np.array(i)
                cur_node['j'] = np.array(j)
                cur_node['density'] = count

                color_index = np.nonzero(
                    np.abs((colors - c)).sum(axis=-1) == 0)[0][0]
                cur_node['c'] = (
                    (tsne_result[color_index][0] - min_tsne) /
                    (max_tsne - min_tsne + 1e-5) * color_unit_size) // 1
                # cur_node['c'] = cluster[color_index]

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
    img = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)

    img_name = image.split('/')[-1].split('.')[0]
    patch_size = 80

    # split the whole image(320*480) into 6 patches (80*80)
    for i in range(320 // patch_size):
        for j in range(480 // patch_size):
            img_patch = img[i * patch_size:(i + 1) * patch_size,
                            j * patch_size:(j + 1) * patch_size]
            gt_patch = gt[i * patch_size:(i + 1) * patch_size,
                          j * patch_size:(j + 1) * patch_size]
            class_index = 1 if 2 in np.unique(gt_patch) else 0
            patch_name = img_name + '_{}_{}'.format(i, j)

            func(img_patch, patch_name, class_index)

            # cv2.imwrite(
            #     os.path.join(datasets_root, dataset_name,|
            #                  class_names[class_index], patch_name + '.jpg'),
            #     img_patch)
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
        os.path.join(datasets_root, dataset_name, 'groundTruth', f'*.mat'))
    img_list = glob.glob(
        os.path.join(datasets_root, dataset_name, 'image', f'*.jpg'))

    para = [(groundTruth, image)
            for groundTruth, image in zip(groundTruth_list, img_list)]
    print(len(para))
    # for gt, img in zip(groundTruth_list, img_list):
    #     img_convert(gt, img)

    pool.starmap_async(img_convert, para)
    pool.close()
    pool.join()
