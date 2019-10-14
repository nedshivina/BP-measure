import torch
import torch.nn as nn
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from torch import autograd
import torch.nn as nn
import torch.optim as optim
import os

feature_sz = 50
max_bp = 180

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(feature_sz, 35)
        self.fc2 = nn.Linear(35, 2)

    def sigmoid(self, s):
        return 1 / (1 + torch.exp(-s))

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        o = x
        return o

def get_train_and_valid_set(feature_path, train_share=0.8, if_random=True):
    feature_x = np.load(feature_path + "/feature_set_x.npy")
    feature_y = np.load(feature_path + "/feature_set_y.npy")

    print(str(len(feature_x)) + " features")

    ids = np.arange(len(feature_x), dtype=int)
    if if_random:
        np.random.shuffle(ids)
    ids = np.array(ids)
    data_set_shuffle = []
    bp_set_shuffle = []

    for id in ids:
        data_set_shuffle.append(feature_x[id])
        bp_set_shuffle.append(feature_y[id])

    data_size = len(data_set_shuffle)

    train_size = int(train_share * data_size)
    train_x = data_set_shuffle[:train_size]
    valid_x = data_set_shuffle[train_size:]

    train_y = bp_set_shuffle[:train_size]
    valid_y = bp_set_shuffle[train_size:]
    print("Random shuffled: " + str(if_random))
    print("Train size " +  str(len(train_y)), "Valid size " + str(len(valid_y)))

    print("Train set BP range: " + str(np.min(train_y)*max_bp) + " " + str(np.max(train_y)*max_bp))
    print("Valid set BP range: " + str(np.min(valid_y)*max_bp) + " " + str(np.max(valid_y)*max_bp))
    train_x = np.float32(train_x)
    train_y = np.float32(train_y)
    valid_x = np.float32(valid_x)
    valid_y = np.float32(valid_y)

    return train_x, train_y, valid_x, valid_y
import sys
import csv
if __name__ == "__main__":

    train = True
    test = True
    once = False

    args = sys.argv
    model_name = args[3]



    train_x, train_y, valid_x, valid_y = get_train_and_valid_set(args[1],if_random=False, train_share=0.8)

    if not os.path.exists(args[2] + model_name):
        os.mkdir(args[2] + model_name)
    print("Model will be saved to " + args[2] + model_name )
    print("Number of epochs " + str(args[4]))
    if train:
        learning_rate = 0.0001
        NN = Net()
        optimizer = optim.SGD(NN.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        batch_size = 1
        n_epoch = int(args[4])
        total_loss_ = np.zeros(n_epoch)
        valid_loss_ = np.zeros(n_epoch)
        epoch_ = 0

        for epoch in range(n_epoch):
            batch_idx = 1
            total_loss = 0
            epoch_ += 1
            for k in range(len(train_x)):
                (data, target) = (torch.from_numpy(np.array(train_x[k])), torch.from_numpy(np.array(train_y[k])))
                data, target = autograd.Variable(data), autograd.Variable(target)
                optimizer.zero_grad()
                net_out = NN(data)
                loss = criterion(net_out, target)
                loss.backward()
                optimizer.step()
                loss_values = loss.item()
                batch_idx += 1
                total_loss += (loss_values)

            total_loss_[epoch] = total_loss * max_bp / len(train_y)
            loss_values1 = 0
            for k in range(len(valid_y)):
                data1 = torch.from_numpy(np.array(valid_x[k]))
                target = torch.from_numpy(np.array(valid_y[k]))
                net_out = NN(data1)
                loss1 = criterion(net_out, target)
                out = net_out.detach().numpy() * max_bp
                target_ = target.detach().numpy() * max_bp
                diff =(abs(target_[0] - out[0]) + abs(target_[1] - out[1]))/2
                loss_values1 += diff.item()

            valid_loss_[epoch] = loss_values1 / len(valid_y)
            if epoch % 10 == 0:
                torch.save(NN, args[2] + model_name + "/" + str(epoch) + ".txt")

            print("Train loss: " + str('% 3.3f' % (total_loss *max_bp/ len(train_y))) + " Valid loss: " + str(epoch) + str('% 3.3f' %valid_loss_[epoch]))
            plt.clf()
            plt.plot(valid_loss_[:epoch_], label='valid_loss')
            plt.legend()
            plt.pause(0.05)
        plt.show()

        plt.clf()
        plt.plot(total_loss_[:epoch_], label='train_loss')
        plt.legend()

        plt.show()
        plt.plot(valid_loss_[:epoch_], label='valid_loss')
        plt.legend()
        plt.show()
        torch.save(NN, args[2] + model_name + ".txt")

    if test:
        NN = torch.load(args[2] + model_name + ".txt")
        print("Validation..")
        print("Target BP.........Model BP")
        loss_values1 = 0

        with open(args[2] + model_name + "/" + "test_results.csv",'w', newline="") as csvfile:
            fieldnames = ['Target_DBP', 'Target_SBP', 'Model_DBP', 'Model_SBP']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for k in range(len(valid_y)):
                data1 = torch.from_numpy(np.array(valid_x[k]))
                net_out = NN(data1).detach().numpy()
                print(valid_y[k] *max_bp, net_out *max_bp)

                writer.writerow({'Target_DBP': valid_y[k][0] *max_bp, 'Target_SBP': valid_y[k][1] *max_bp, 'Model_DBP': net_out[0] *max_bp, 'Model_SBP':net_out[1] *max_bp})




