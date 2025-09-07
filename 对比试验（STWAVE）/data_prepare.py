import torch
import numpy as np
import os
from utils import print_log, StandardScaler, vrange

# ! X shape: (B, T, N, C)


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

    num_step = data.shape[0]

    TE = np.zeros([num_step, 2])

    TE[:, 1] = np.array([i % 288 for i in range(num_step)])
    TE[:, 0] = np.array([(i // 288) % 7 for i in range(num_step)])

    TE_tile = np.repeat(np.expand_dims(TE, 1), data.shape[1], 1)




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
    y_train = data[y_train_index][..., :1]

    trainXTE=TE_tile[x_train_index]

    trainYTE=TE_tile[y_train_index]
    #y_train = data[y_train_index]
    x_val = data[x_val_index]
    y_val = data[y_val_index][..., :1]
    valXTE = TE_tile[x_val_index]
    valYTE = TE_tile[y_val_index]
    #y_val = data[y_val_index]
    x_test = data[x_test_index]
    y_test = data[y_test_index][..., :1]
    testXTE = TE_tile[x_test_index]
    testYTE = TE_tile[y_test_index]
   # y_test = data[y_test_index]

    trainTE = np.concatenate([trainXTE, trainYTE], axis=1)

    valTE = np.concatenate([valXTE, valYTE], axis=1)
    testTE = np.concatenate([testXTE, testYTE], axis=1)

    scaler = StandardScaler(mean=x_train[..., 0].mean(), std=x_train[..., 0].std())

    x_train[..., 0] = scaler.transform(x_train[..., 0])
    x_val[..., 0] = scaler.transform(x_val[..., 0])
    x_test[..., 0] = scaler.transform(x_test[..., 0])

    print_log(f"Trainset:\tx-{x_train.shape}\ty-{y_train.shape}", log=log)
    print_log(f"Valset:  \tx-{x_val.shape}  \ty-{y_val.shape}", log=log)
    print_log(f"Testset:\tx-{x_test.shape}\ty-{y_test.shape}", log=log)

    trainset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_train), torch.FloatTensor(y_train), torch.FloatTensor(trainTE)
    )
    valset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_val), torch.FloatTensor(y_val),torch.FloatTensor(valTE)
    )
    testset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_test), torch.FloatTensor(y_test),torch.FloatTensor(testTE)
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
