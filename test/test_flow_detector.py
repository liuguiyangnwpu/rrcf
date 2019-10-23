# -*- coding: utf-8 -*-

import os
import dill
import timeit
import pandas as pd
from collections import deque
import matplotlib.pyplot as plt
from multiprocessing import Pool, Manager
from sklearn.ensemble import IsolationForest
from rrcf import RCTree


# Read data
def data_explore(is_show=True):
    abs_path = os.path.abspath(__file__)
    data_paths = abs_path.strip().split("/")[:-1]
    data_paths.extend(["..", "resources", "website_flow.csv"])
    data_path = os.path.join(*data_paths)
    if not data_path.startswith("/"):
        data_path = "/" + data_path
    data_frame = pd.read_csv(data_path)
    data_frame.sort_values(["time"], inplace=True)
    data_frame["time"] = pd.to_datetime(data_frame["time"], unit="s")
    data_frame.set_index(["time"], inplace=True)
    print(data_frame.head(3))

    if is_show:
        fig, ax = plt.subplots(3, figsize=(12, 7))
        data_frame["bytes"].plot(ax=ax[0])
        data_frame["request"].plot(ax=ax[1])
        data_frame["num"].plot(ax=ax[2])
        ax[0].set_ylabel("bytes")
        ax[1].set_ylabel("request")
        ax[2].set_ylabel("num")
        plt.tight_layout()
        plt.show()

    return data_frame


def __create_tree__(tree_id, data_frame, result_dict):
    tree = RCTree(data_frame.values, index_labels=list(data_frame.index))
    tree_dumps = dill.dumps(tree)
    result_dict[tree_id] = tree_dumps


def benchmark_create_rrcf():
    num_trees = 5
    def single_func(tree_id, data_frame):
        tree = RCTree(data_frame.values, index_labels=list(data_frame.index))

    start = timeit.default_timer()
    for idx in range(num_trees):
        single_func(idx, history_samples)
    stop = timeit.default_timer()
    print("total spend is {}; avg spend is {}".format(stop-start, (stop-start)/num_trees))


def __calc_avg_codisp_socre__(tree_id, tree_dump, data_frame, result_dict):
    tree = dill.loads(tree_dump)
    print("tree_id is {}".format(tree_id))
    anomaly_score = pd.Series(0.0, index=data_frame.index)
    for idx in data_frame.index:
        anomaly_score[idx] = tree.codisp(idx)
    result_dict[tree_id] = anomaly_score


# Set tree parameters
num_trees = 100
tree_size = 6000
history_queue = deque([], maxlen=tree_size)

n = 5000
data_frame = data_explore(is_show=False)
history_samples = data_frame[:n]
testing_samples = data_frame[n:]


def flow_detector_by_isolate_forest():
    # fit the model
    clf = IsolationForest(n_estimators=100,
                          behaviour='new',
                          max_samples=n*2,
                          contamination='auto',
                          n_jobs=1,
                          verbose=1)
    clf.fit(history_samples.values)
    pred = clf.predict(history_samples.values)
    print(pred)


def flow_detector():
    anomaly_score = pd.Series(0.0, index=data_frame.index)

    start = timeit.default_timer()
    manager = Manager()
    return_dict = manager.dict()
    pool = Pool(processes=5)
    for tree_id in range(num_trees):
        pool.apply_async(func=__create_tree__, args=(tree_id, history_samples, return_dict, ))
    pool.close()
    pool.join()
    forest = [dill.loads(obj) for obj in return_dict.values()]
    stop = timeit.default_timer()
    print("forest length is {}, Time: {}".format(len(forest), stop-start))
    return

    manager_history = Manager()
    return_history_dict = manager_history.dict()
    pool_history = Pool(processes=5)
    for idx, tree in enumerate(forest):
        tree_dump = dill.dumps(tree)
        pool_history.apply_async(func=__calc_avg_codisp_socre__, args=(idx, tree_dump, history_samples, return_history_dict,))
    pool_history.close()
    pool_history.join()

    for obj in return_history_dict.values():
        anomaly_score += obj
    anomaly_score = anomaly_score / num_trees

    for idx in list(history_samples.index):
        history_queue.append(idx)

    for idx in testing_samples.index:
        cur_point = testing_samples.ix[idx].values
        if len(history_queue) == tree_size:
            old_index = history_queue.popleft()
        else:
            old_index = None
        history_queue.append(idx)

        avg_codisp = 0.0
        for tree in forest:
            if old_index is not None:
                tree.forget_point(old_index)
            tree.insert_point(point=cur_point, index=idx)
            avg_codisp += tree.codisp(idx) / num_trees
        # print('CoDisp for point ({index}) is {avg_codisp}'.format(index=idx, avg_codisp=avg_codisp))
        anomaly_score[idx] = avg_codisp

    if True:
        fig, ax = plt.subplots(4, figsize=(20, 8))
        data_frame["bytes"].plot(ax=ax[0])
        data_frame["request"].plot(ax=ax[1])
        data_frame["num"].plot(ax=ax[2])
        anomaly_score.plot(ax=ax[3])

        ax[0].set_ylabel("bytes")
        ax[1].set_ylabel("request")
        ax[2].set_ylabel("num")
        ax[3].set_ylabel("codisp_score")
        plt.tight_layout()
        plt.show()


from rrcf.crcf import RobustRandomCutTree
def test_crcf():
    start = timeit.default_timer()
    for i in range(num_trees):
        obj = RobustRandomCutTree(depth_limit=100)
        obj.fit(history_samples.values)
        pred = obj.score(history_samples.values)
    stop = timeit.default_timer()
    print("total spend is {}, avg spend is {}".format(stop-start, (stop-start)/num_trees))



if __name__ == '__main__':
    # flow_detector()
    # flow_detector_by_isolate_forest()
    benchmark_create_rrcf()
    # test_crcf()