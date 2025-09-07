import torch
import numpy as np
import os
from .utils import print_log, StandardScaler, vrange
import pandas as pd

# ! X shape: (B, T, N, C)

import matplotlib.pyplot as plt
def get_dataloaders_from_index_data(
    data_dir, tod=False, dow=False, dom=False, batch_size=64, log=None
):
    data = np.load(os.path.join(data_dir, "data.npz"))["data"].astype(np.float32)

    features = [0]
    if tod:
        features.append(1)
    if dow:
        features.append(2)
    # if dom:
    #     features.append(3)
    data = data[..., features]



    index = np.load(os.path.join(data_dir, "index.npz"))

    train_index = index["train"]  # (num_samples, 3)
    val_index = index["val"]
    test_index = index["test"]

    x_train_index = vrange(train_index[:, 0], train_index[:, 1])
    y_train_index = vrange(train_index[:, 1], train_index[:, 2])
    x_val_index = vrange(val_index[:, 0], val_index[:, 1])
    y_val_index = vrange(val_index[:, 1], val_index[:, 2])
    x_test_index = vrange(test_index[:, 0], test_index[:, 1])
    y_test_index = vrange(test_index[:, 1], test_index[:, 2])

    x_train = data[x_train_index]
    """



    time_interval = pd.date_range(start='00:00', end='23:55', freq='5T')
    #hours = time_interval.hour + time_interval.minute / 60  # 转换为小时表示
    hours = time_interval.strftime('%H:%M')

    plt.figure(figsize=(12, 6))

    #plt.xticks(np.arange(0, 25, 1))
    plt.xticks(ticks=np.arange(0, len(hours), 12),
               labels=hours[::12], rotation=45)  # 每12个5分钟显示一次（即每小时显示一次）
    plt.xticks(list(plt.xticks()[0]) + [len(hours) - 1],
               list(plt.xticks()[1]) + ['24:00'], rotation=45)
               
    


    x_pig1=  data[:800,14,0]
    x_pig1=x_pig1[3::4]

    #x_pig2 = data[:12,0,0]
    #x_pig3 = data[:12,15,0]

    plt.plot(np.arange(0,x_pig1.shape[0] ), x_pig1, 'b-', alpha=0.5, linewidth=1, label='Variable 1')
    #plt.plot(np.arange(0, x_pig1.shape[0]), x_pig2, 'r-', alpha=0.5, linewidth=1, label='Variable 1')
    #plt.plot(np.arange(0, x_pig1.shape[0]), x_pig3, 'g-', alpha=0.5, linewidth=1, label='Variable 1')



    #plt.plot(hours, x_pig1, 'b*--', alpha=0.5, linewidth=1, label='Area A')  # '
    #plt.plot(hours, x_pig2, 'rs--', alpha=0.5, linewidth=1, label='Area B')
    #plt.plot(hours, x_pig3, 'go--', alpha=0.5, linewidth=1, label='Area C')

    #plt.grid()
    #plt.legend()  # 显示上面的label
    plt.xlabel('Time')
    plt.ylabel('Flow')  # accuracy

    # 显示图形
    plt.tight_layout()
    #plt.savefig(os.path.join(f'../fig' , 'Flow.png'))
    plt.show()
    """







    y_train = data[y_train_index][..., :1]
    #y_train = data[y_train_index]
    x_val = data[x_val_index]
    y_val = data[y_val_index][..., :1]
    #y_val = data[y_val_index]
    x_test = data[x_test_index]
    y_test = data[y_test_index][..., :1]
   # y_test = data[y_test_index]

    scaler = StandardScaler(mean=x_train[..., 0].mean(), std=x_train[..., 0].std())

    x_train[..., 0] = scaler.transform(x_train[..., 0])
    x_val[..., 0] = scaler.transform(x_val[..., 0])
    x_test[..., 0] = scaler.transform(x_test[..., 0])

    print_log(f"Trainset:\tx-{x_train.shape}\ty-{y_train.shape}", log=log)
    print_log(f"Valset:  \tx-{x_val.shape}  \ty-{y_val.shape}", log=log)
    print_log(f"Testset:\tx-{x_test.shape}\ty-{y_test.shape}", log=log)

    trainset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_train), torch.FloatTensor(y_train)
    )
    valset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_val), torch.FloatTensor(y_val)
    )
    testset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_test), torch.FloatTensor(y_test)
    )

    trainset_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True
    )
    valset_loader = torch.utils.data.DataLoader(
        valset, batch_size=batch_size, shuffle=False
    )
    testset_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False
    )

    return trainset_loader, valset_loader, testset_loader, scaler
