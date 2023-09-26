import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def dirichlet_split(x, y, client_nums, class_nums,train_size):
    client_datas = dict()
    min_size = 0
    min_require_size = 30
    N = y.shape[0]
    idx_batch = []
    while min_size < min_require_size:
        idx_batch = [[] for _ in range(client_nums)]
        for k in range(class_nums):
            idx_k = np.where(y == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(0.5, client_nums))
            proportions = np.array([p * (len(idx_j) < N / client_nums) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(client_nums):
        sample = x[idx_batch[j]]
        label = y[idx_batch[j]]

        state = np.random.get_state()
        np.random.shuffle(sample)
        np.random.set_state(state)
        np.random.shuffle(label)

        train_data, test_data, train_labels, test_labels = train_test_split(sample, label, test_size=1-train_size, random_state=42, )

        if len(train_data.shape) ==2:
            shape = (train_data.shape[0], train_data.shape[1], 1)
        else:shape = train_data.shape
        scaler = MinMaxScaler()
        train_data = scaler.fit_transform(train_data.reshape((-1, shape[-1]))).reshape(shape)
        test_data = scaler.transform(test_data.reshape((-1, shape[-1]))).reshape((-1, shape[1], shape[-1]))
        client_datas[j] = {"x_train": train_data, "y_train": train_labels,"x_test":test_data,"y_test":test_labels}

    return client_datas
