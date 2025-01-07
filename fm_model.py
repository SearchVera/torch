# 196	242	3	881250949
# 186	302	3	891717742

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

learning_rate = 1e-3
weight_decay = 1e-4
epochs = 100
batch_size = 1000
min_val, max_val = 1.0, 5.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
id_embedding_dim = 256


class FMLayer(nn.Module):
    def __init__(self, p, k):
        super(FMLayer, self).__init__()
        self.p, self.k = p, k

        self.linear = nn.Linear(self.p, 1, bias=True)
        # [dim*2, 10]
        self.v = nn.Parameter(torch.Tensor(self.p, self.k), requires_grad=True)
        self.v.data.uniform_(-0.01, 0.01)
        self.drop = nn.Dropout(0.2)

    def forward(self, x):
        '''

        :param x: [batch_size, dim*2]
        :return:
        '''
        # [batch_size, 1]
        linear_part = self.linear(x)
        # [batch_size, 10]
        inter_part1 = torch.pow(torch.mm(x, self.v), 2)
        # [batch_size, 10]
        inter_part2 = torch.mm(torch.pow(x, 2), torch.pow(self.v, 2))
        # [batch_size]
        pair_interactions = torch.sum(torch.sub(inter_part1, inter_part2), dim=1)
        pair_interactions = self.drop(pair_interactions)
        # [1, batch_size]
        output = linear_part.transpose(1, 0) + 0.5 * pair_interactions
        # [batch_size, 1]
        return output.view(-1, 1)


class FM(nn.Module):
    def __init__(self, user_nums, item_nums, id_embedding_dim):
        super(FM, self).__init__()

        self.user_id_vec = nn.Embedding(user_nums, id_embedding_dim)
        self.item_id_vec = nn.Embedding(item_nums, id_embedding_dim)

        self.fm = FMLayer(id_embedding_dim * 2, 10)

    def forward(self, user_id, item_id):
        u_vec = self.user_id_vec(user_id)
        i_vec = self.item_id_vec(item_id)
        x = torch.cat((u_vec, i_vec), dim=1)
        rate = self.fm(x)
        return rate


class FMDataset(Dataset):
    def __init__(self, uid, iid, rating):
        self.uid = uid
        self.iid = iid
        self.rating = rating

    def __getitem__(self, index):
        return self.uid[index], self.iid[index], self.rating[index]

    def __len__(self):
        return len(self.uid)


def train_iter(model, optimizer, train_loader, criterion):
    model.train()
    total_loss, total_len = 0, 0

    for index, (x_u, x_i, y) in enumerate(train_loader):
        x_u, x_i, y = x_u.to(device), x_i.to(device), y.to(device)
        y = (y - min_val) / (max_val - min_val) + 0.01
        y_pred = model(x_u, x_i)

        loss = criterion(y.view(-1, 1), y_pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y_pred)
        total_len += len(y_pred)

    return total_loss / total_len

def val_iter(model, dataloader):
    model.eval()
    labels, predicts = list(), list()
    with torch.no_grad():
        for x_u, x_i, y in dataloader:
            x_u, x_i, y = x_u.to(device), x_i.to(device), y.to(device)
            y_pred = model(x_u, x_i)
            y_pred = min_val + (y_pred - 0.01) * (max_val - min_val)
            y_pred = torch.where(y_pred>max_val, torch.full_like(y_pred, max_val), y_pred)
            y_pred = torch.where(y_pred<min_val, torch.full_like(y_pred, min_val), y_pred)
            labels.extend(y)
            predicts.extend(y_pred)
    return mean_squared_error(np.array(labels), np.array(predicts))



def main():
    df = pd.read_csv('./data/u.data', header=None, delimiter='\t')
    u_max_id = max(df[0]) + 1
    i_max_id = max(df[1]) + 1
    print(df.shape[0], u_max_id, i_max_id)

    x, y = df.iloc[:, :2], df.iloc[:, 2]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    train_loader = DataLoader(
        FMDataset(np.array(x_train[0]), np.array(x_train[1]), np.array(y_train).astype(np.float32)),
        batch_size=batch_size)
    val_loader = DataLoader(FMDataset(np.array(x_val[0]), np.array(x_val[1]), np.array(y_val).astype(np.float32)),
                            batch_size=batch_size)
    test_loader = DataLoader(FMDataset(np.array(x_test[0]), np.array(x_test[1]), np.array(y_test).astype(np.float32)),
                             batch_size=batch_size)

    model = FM(u_max_id, i_max_id, id_embedding_dim).to(device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_func = torch.nn.MSELoss().to(device)

    best_val_mse, best_val_epoch = 10, 1
    for epoch in range(epochs):
        loss = train_iter(model, optimizer, train_loader, loss_func)
        mse = val_iter(model, val_loader)
        print("epoch:{}, loss:{:.5}, mse:{:.5}".format(epoch, loss, mse))
        if best_val_mse > mse:
            best_val_mse, best_val_epoch = mse, epoch
            torch.save(model, 'best_model')
    print("best val epoch is {}, mse is {}".format(best_val_epoch, best_val_mse))
    model = torch.load('best_model').to(device)
    test_mse = val_iter(model, test_loader)
    print(f'test mse: {test_mse}')


if __name__ == '__main__':
    main()
