import scipy.io
import numpy as np
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from . import data_split


def get_tensors_from_matlab(matlab_files_name):
    acquisitions = {}
    for key in matlab_files_name:
        file_name = matlab_files_name[key]
        matlab_file = scipy.io.loadmat(file_name)
        for position in ['DE']:
            keys = [key for key in matlab_file if key.endswith(position + "_time")]
            if len(keys) > 0:
                array_key = keys[0]
                acquisitions[key] = matlab_file[array_key].reshape(1, -1)[0]
    return acquisitions


def CWRU_segmentation(acquisitions, sample_size=512, max_samples=None):
    origin = []
    data = np.empty((0, sample_size, 1))
    n = len(acquisitions)
    for i, key in enumerate(acquisitions):
        acquisition_size = len(acquisitions[key])
        n_samples = acquisition_size // sample_size
        if max_samples is not None and max_samples > 0 and n_samples > max_samples:
            n_samples = max_samples
        origin.extend([key for _ in range(n_samples)])
        data = np.concatenate((data,
                               acquisitions[key][:(n_samples * sample_size)].reshape(
                                   (n_samples, sample_size, 1))))
    return data, origin


def select_samples(regex, X, y):
    mask = [re.search(regex, label) is not None for label in y]
    return X[mask], y[mask]


def join_labels(regex, y):
    return np.array([re.sub(regex, '', label) for label in y])


def samples_relabel(regex, rp, y):
    mask = [re.search(regex, label) is not None for label in y]
    y[mask] = rp
    return y


def get_groups(regex, y):
    groups = list(range(len(y)))
    for i, label in enumerate(y):
        match = re.search(regex, label)
        groups[i] = match.group(0) if match else None
    return groups


def CWRU_12khz():
    matlab_files_name = {}
    # Normal
    matlab_files_name["Normal_0"] = "datasets/CWRU/97.mat"
    matlab_files_name["Normal_1"] = "datasets/CWRU/98.mat"
    matlab_files_name["Normal_2"] = "datasets/CWRU/99.mat"
    matlab_files_name["Normal_3"] = "datasets/CWRU/100.mat"
    # DE Inner Race 0.007 inches
    matlab_files_name["DEIR007_0"] = "datasets/CWRU/105.mat"
    matlab_files_name["DEIR007_1"] = "datasets/CWRU/106.mat"
    matlab_files_name["DEIR007_2"] = "datasets/CWRU/107.mat"
    matlab_files_name["DEIR007_3"] = "datasets/CWRU/108.mat"
    # DE Inner Race 0.014 inches
    matlab_files_name["DEIR014_0"] = "datasets/CWRU/169.mat"
    matlab_files_name["DEIR014_1"] = "datasets/CWRU/170.mat"
    matlab_files_name["DEIR014_2"] = "datasets/CWRU/171.mat"
    matlab_files_name["DEIR014_3"] = "datasets/CWRU/172.mat"
    #DE Inner Race 0.021 inches
    matlab_files_name["DEIR021_0"] = "datasets/CWRU/209.mat"
    matlab_files_name["DEIR021_1"] = "datasets/CWRU/210.mat"
    matlab_files_name["DEIR021_2"] = "datasets/CWRU/211.mat"
    matlab_files_name["DEIR021_3"] = "datasets/CWRU/212.mat"


    # DE Ball 0.007 inches
    matlab_files_name["DEB007_0"] = "datasets/CWRU/118.mat"
    matlab_files_name["DEB007_1"] = "datasets/CWRU/119.mat"
    matlab_files_name["DEB007_2"] = "datasets/CWRU/120.mat"
    matlab_files_name["DEB007_3"] = "datasets/CWRU/121.mat"
    # DE Ball 0.014 inches
    matlab_files_name["DEB014_0"] = "datasets/CWRU/185.mat"
    matlab_files_name["DEB014_1"] = "datasets/CWRU/186.mat"
    matlab_files_name["DEB014_2"] = "datasets/CWRU/187.mat"
    matlab_files_name["DEB014_3"] = "datasets/CWRU/188.mat"
    #DE Ball 0.021 inches
    matlab_files_name["DEB021_0"] = "datasets/CWRU/222.mat"
    matlab_files_name["DEB021_1"] = "datasets/CWRU/223.mat"
    matlab_files_name["DEB021_2"] = "datasets/CWRU/224.mat"
    matlab_files_name["DEB021_3"] = "datasets/CWRU/225.mat"


    # DE Outer race 0.007 inches centered @6:00
    matlab_files_name["DEOR@6_007_0"] = "datasets/CWRU/130.mat"
    matlab_files_name["DEOR@6_007_1"] = "datasets/CWRU/131.mat"
    matlab_files_name["DEOR@6_007_2"] = "datasets/CWRU/132.mat"
    matlab_files_name["DEOR@6_007_3"] = "datasets/CWRU/133.mat"
    #DE Outer race 0.014 inches centered @6:00
    matlab_files_name["DEOR@6_014_0"] = "datasets/CWRU/197.mat"
    matlab_files_name["DEOR@6_014_1"] = "datasets/CWRU/198.mat"
    matlab_files_name["DEOR@6_014_2"] = "datasets/CWRU/199.mat"
    matlab_files_name["DEOR@6_014_3"] = "datasets/CWRU/200.mat"
    #DE Outer race 0.021 inches centered @6:00
    matlab_files_name["DEOR@6_021_0"] = "datasets/CWRU/234.mat"
    matlab_files_name["DEOR@6_021_1"] = "datasets/CWRU/235.mat"
    matlab_files_name["DEOR@6_021_2"] = "datasets/CWRU/236.mat"
    matlab_files_name["DEOR@6_021_3"] = "datasets/CWRU/237.mat"

    return matlab_files_name

#5926
def get_data(time_length, client_nums,train_size):
    acquisitions = get_tensors_from_matlab(CWRU_12khz())
    signal_data, signal_origin = CWRU_segmentation(acquisitions, time_length)
    relabel_signal = np.array(signal_origin)
    regex = '^(Normal).*'
    rp = 'Normal'
    relabel_signal = samples_relabel(regex, rp, relabel_signal)

    regex = '^(DEIR007).*'
    rp = 'DEIR007'
    relabel_signal = samples_relabel(regex, rp, relabel_signal)
    regex = '^(DEIR014).*'
    rp = 'DEIR014'
    relabel_signal = samples_relabel(regex, rp, relabel_signal)
    regex = '^(DEIR021).*'
    rp = 'DEIR021'
    relabel_signal = samples_relabel(regex, rp, relabel_signal)

    regex = '^(DEB007).*'
    rp = 'DEB007'
    relabel_signal = samples_relabel(regex, rp, relabel_signal)
    regex = '^(DEB014).*'
    rp = 'DEB014'
    relabel_signal = samples_relabel(regex, rp, relabel_signal)
    regex = '^(DEB021).*'
    rp = 'DEB021'
    relabel_signal = samples_relabel(regex, rp, relabel_signal)


    regex = '^(DEOR@6_007).*'
    rp = 'DEOR007'
    relabel_signal = samples_relabel(regex, rp, relabel_signal)
    regex = '^(DEOR@6_014).*'
    rp = 'DEOR014'
    relabel_signal = samples_relabel(regex, rp, relabel_signal)
    regex = '^(DEOR@6_021).*'
    rp = 'DEOR021'
    relabel_signal = samples_relabel(regex, rp, relabel_signal)



    samples = '^(DE)|(Normal).*'
    X, y = select_samples(samples, signal_data, relabel_signal)

    labels = pd.Categorical(y, categories=set(y)).codes

    mapping =  pd.Categorical.from_codes([x for x in range(10)], categories=set(y)).categories


    client_datas = data_split.dirichlet_split(X, labels, client_nums, len(set(labels)),train_size)
    #
    return client_datas,mapping



