import pandas as pd
import numpy as np
def load_dataset():
    headers = ["unit number","cycles","operational setting 1","operational setting 2","operational setting 3"]
    [headers.append("sensor"+str(x)) for x in range(1,22)]
    train = pd.read_csv('datasets/CMAPSS/train_FD001.txt',sep=" ",skipinitialspace=True,header=None)
    test = pd.read_csv('datasets/CMAPSS/test_FD001.txt',sep=" ",skipinitialspace=True,header=None)
    test_rul = pd.read_csv('datasets/CMAPSS/RUL_FD001.txt',sep=" ",skipinitialspace=True,header=None)

    train = train.drop(list(train)[26],axis=1)
    test = test.drop(list(test)[26],axis=1)
    test_rul = test_rul.drop(list(test_rul)[1],axis=1)

    train.columns = tuple(headers)
    test.columns = tuple(headers)
    test_rul.columns=('remaining',)
    #test.insert(26, "Target_Remaining_Useful_Life", test_rul,allow_duplicates = False)

    train=calcu_train_hi(train)
    test=calcu_test_hi(test,test_rul)
    #test["Target_Remaining_Useful_Life"] =  test["Target_Remaining_Useful_Life"].apply(hi)


    train["unit number"] = train['unit number'].astype(str).astype("category")
    test["unit number"] = test['unit number'].astype(str).astype("category")
    return train,test

def hi(x):
    return 1-np.exp((1-x)/45)
def calcu_train_hi(train):
    max_cycle = train.groupby('unit number')['cycles'].max().reset_index()
    max_cycle.columns = ['unit number', 'MaxOfCycle']
    # merge the max cycle back into the original frame
    train_merged = train.merge(max_cycle, left_on='unit number', right_on='unit number', how='inner')
    # calculate RUL for each row
    Target_Remaining_Useful_Life = train_merged["MaxOfCycle"] - train_merged["cycles"]
    train_with_target = train_merged["Target_Remaining_Useful_Life"] = Target_Remaining_Useful_Life
    # remove unnecessary column
    train_with_target = train_merged.drop("MaxOfCycle", axis=1)
    train_with_target["Target_Remaining_Useful_Life"] =train_with_target["Target_Remaining_Useful_Life"].apply(hi)
    return train_with_target
def calcu_test_hi(test,rul):
    max_cycle = test.groupby('unit number')['cycles'].max().reset_index()
    max_cycle.columns = ['unit number', 'MaxOfCycle']

    rul.insert(0, "unit number", [x for x in range(1,101)],allow_duplicates = False)

    test_merged = test.merge(max_cycle, left_on='unit number', right_on='unit number', how='inner')
    test_merged = test_merged.merge(rul, left_on='unit number', right_on='unit number', how='inner')



    Target_Remaining_Useful_Life = test_merged["MaxOfCycle"] - test_merged["cycles"] +test_merged["remaining"]
    test_merged["Target_Remaining_Useful_Life"] = Target_Remaining_Useful_Life
    test_merged = test_merged.drop("MaxOfCycle", axis=1)
    test_merged = test_merged.drop("remaining", axis=1)
    test_merged["Target_Remaining_Useful_Life"] =test_merged["Target_Remaining_Useful_Life"].apply(hi)
    return test_merged