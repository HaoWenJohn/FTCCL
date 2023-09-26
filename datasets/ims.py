from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
from . import data_split
import pandas as pd

# label strategy : health=0,degradation=1,inner_race_defect=2,roller_element_defect=3mouter_race_defect=4
def get_data(time_length, client_nums, train_size):
    dir_path = 'datasets/IMS/'

    health_rate = 0.2
    degradation_rate = 0.6

    exp1 = np.load(dir_path + "exp1.npy")
    exp2 = np.load(dir_path + "exp2.npy")
    exp3 = np.load(dir_path + "exp3.npy")

    inner_race_defect = exp1[:, :, 2]
    roller_element_defect = exp1[:, :, 3]
    outer_race_defect = np.vstack([exp2[:, :, 0], exp3[:, :, 2]])

    normal = np.concatenate([exp1[:, :, 0], exp1[:, :, 1], exp2[:, :, 1], exp2[:, :, 2], exp2[:, :, 3]
                                , exp3[:, :, 0], exp3[:, :, 1], exp3[:, :, 3]])

    samples = []
    labels = []

    nomal_health = int(normal.shape[0] * health_rate)
    nomal_degradation = normal.shape[0] - nomal_health
    samples.append(normal[0:nomal_health])
    labels.append(np.zeros(nomal_health))
    samples.append(normal[nomal_health:])
    labels.append(np.ones(nomal_degradation))

    inner_race_defect_health = int(inner_race_defect.shape[0] * health_rate)
    inner_race_defect_degradation = int(inner_race_defect.shape[0] * degradation_rate)
    inner_race_defect_failure = inner_race_defect.shape[0] - inner_race_defect_health - inner_race_defect_degradation
    samples.append(inner_race_defect[0:inner_race_defect_health])
    labels.append(np.zeros(inner_race_defect_health))
    samples.append(inner_race_defect[inner_race_defect_health:inner_race_defect_health + inner_race_defect_degradation])
    labels.append(np.ones(inner_race_defect_degradation))
    samples.append(inner_race_defect[inner_race_defect_health + inner_race_defect_degradation:])
    tmp = np.empty(inner_race_defect_failure)
    tmp.fill(2)
    labels.append(tmp)

    roller_element_defect_health = int(roller_element_defect.shape[0] * health_rate)
    roller_element_defect_degradation = int(roller_element_defect.shape[0] * degradation_rate)
    roller_element_defect_failure = roller_element_defect.shape[
                                        0] - roller_element_defect_health - roller_element_defect_degradation
    samples.append(roller_element_defect[0:roller_element_defect_health])
    labels.append(np.zeros(roller_element_defect_health))
    samples.append(roller_element_defect[
                   roller_element_defect_health:roller_element_defect_health + roller_element_defect_degradation])
    labels.append(np.ones(roller_element_defect_degradation))
    samples.append(roller_element_defect[roller_element_defect_health + roller_element_defect_degradation:])
    tmp = np.empty(roller_element_defect_failure)
    tmp.fill(3)
    labels.append(tmp)

    outer_race_defect_health = int(outer_race_defect.shape[0] * health_rate)
    outer_race_defect_degradation = int(outer_race_defect.shape[0] * degradation_rate)
    outer_race_defect_failure = outer_race_defect.shape[0] - outer_race_defect_health - outer_race_defect_degradation
    samples.append(outer_race_defect[0:outer_race_defect_health])
    labels.append(np.zeros(outer_race_defect_health))
    samples.append(outer_race_defect[outer_race_defect_health:outer_race_defect_health + outer_race_defect_degradation])
    labels.append(np.ones(outer_race_defect_degradation))
    samples.append(outer_race_defect[outer_race_defect_health + outer_race_defect_degradation:])
    tmp = np.empty(outer_race_defect_failure)
    tmp.fill(4)
    labels.append(tmp)

    x = np.concatenate(samples)
    labels = np.concatenate(labels)

    x_split = np.empty((x.shape[0], time_length))
    split_end = x.shape[1] - time_length
    for i in range(x.shape[0]):
        split_idx = np.random.randint(0, split_end)
        x_split[i] = x[i][split_idx:split_idx + time_length]

    labels = pd.Categorical(labels, categories=set(labels)).codes
    client_datas = data_split.dirichlet_split(x_split, labels, client_nums, 5,train_size)
    return client_datas
