import argparse
import math
import os

import numpy as np

import draw_comfusion_matrix
from n_iid_spilt import draw_heatmap, matrix
import tasks
from utils import init_dl_program
from ftccl import FTCCL

import datasets
import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('epoch',type=int)
    parser.add_argument('global_epoch',type=int)
    parser.add_argument('ws',type=int)
    parser.add_argument('cs',type=int)
    parser.add_argument("train_size",type=float)
    parser.add_argument("fine_tune_size",type=float)
    args = parser.parse_args()


    dataset =args.dataset

    window_size = args.ws

    client_num = args.cs

    local_epoch =args.epoch
    global_epoch =args.global_epoch

    train_size = args.train_size
    fine_tune_size = args.fine_tune_size

    draw_tsne = True
    device = init_dl_program(0, max_threads=8)
    config = dict(
        batch_size=32,
        lr=0.001,
        output_dims=320,
        max_train_length=3000
    )

    run_dir = f'training/{dataset}'

    get_data = None

    if dataset == "CWRU":
        get_data = datasets.cwru_get_data

    elif dataset == "IMS":
        get_data = datasets.ims_get_data

    elif dataset =="MM":
        get_data = datasets.mm_get_data

    else:
        print("unsupported dataset")
        exit(0)
    client_datas,mapping = get_data(window_size, client_num,train_size)

    model = FTCCL(
        input_dims=client_datas[0]["x_train"].shape[-1],
        device=device,
        **config
    )

    server_model_dict = model.net.state_dict()
        #model._net.state_dict()

    for r in range(global_epoch):
        sum_model_weight = None

        for idx, key in enumerate(client_datas.keys()):
            model.net.load_state_dict(server_model_dict, strict=True)

            model.n_epochs = 0
            model.n_iters = 0
            loss_log = model.client_fit(
                client_datas[key]["x_train"],
                n_epochs=local_epoch,
                #n_iters=len(client_datas[key]["x_train"])//client_num//64,
                verbose=True
            )

            client_model_weight = model.net.state_dict()

            if sum_model_weight is None:
                sum_model_weight = client_model_weight
            else:
                for var in sum_model_weight:
                    sum_model_weight[var] = sum_model_weight[var] + client_model_weight[var]

        for var in sum_model_weight:
            sum_model_weight[var] = (sum_model_weight[var] / client_num)

        model.net.load_state_dict(sum_model_weight, strict=True)
        #model.net.update_parameters(model._net)

    avg_acc = 0
    total_confusion = []
    for key in client_datas.keys():
        res,acc,co,confusion = tasks.eval_classification(model,client_datas[key]["x_test"],client_datas[key]["y_test"],fine_tune_size)
        mf1 = res["weighted avg"]["f1-score"]
        confusion = confusion / np.sum(confusion, axis=0)
        for i in  range(confusion.shape[0]):
            for j in range(confusion.shape[1]):
                if  math.isnan(confusion[i][j]):
                    if i ==j:  confusion[i][j] = 1
                    else:confusion[i][j] = 0

        avg_acc+=acc
        total_confusion.append(np.transpose(confusion))
    save_dir = f'{run_dir}_acc_{avg_acc/client_num}_ts_{train_size}_fs_{fine_tune_size}_c_{client_num}'
    os.mkdir(save_dir)
    lab = [x for x in range(10)]
    for idx,c in enumerate(total_confusion):
        draw_comfusion_matrix.draw(lab,c,save_path=f"{save_dir}/{idx}.pdf")
        model.save(f'{save_dir}/weight.pt')
    draw_heatmap(matrix(client_datas,10),f'{save_dir}/CWRU_{client_num}.pdf')
