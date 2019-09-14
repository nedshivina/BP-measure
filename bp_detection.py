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


def process_seq(seq, filename, bp_start, ppg_start, ppg_len, ppg_gt_len, bp_max, data_version):
    data_set = []
    bp_set = []

    if data_version == 3:
        bp_str = filename[bp_start:]
        abp1 = np.float32(bp_str[:3])
        abp2 = np.float32(bp_str[4:])

        for i in range(ppg_start, len(seq), ppg_len):
            seq1 = seq[i:i + ppg_len]
            seq1 = signal.resample(seq1, ppg_gt_len)
            fft_ppg = np.abs(np.fft.fft(seq1))
            if len(seq1) == ppg_gt_len:
                sample = np.zeros(feature_sz)
                sample[0:feature_sz] = abs(fft_ppg[1:feature_sz + 1])
                data_set.append(sample)
                bp_set.append([abp1 / bp_max, abp2 / bp_max])
        return data_set, bp_set


def get_bp_v2(filename):


    bp_str = filename[4:10]
    abp1 = np.float32(bp_str[:3])
    abp2 = np.float32(bp_str[4:])

    return abp1, abp2
def get_our_v2(pulses_path):


    files = os.listdir(pulses_path)
    train_sample = []
    train_y = []
    for file in files:
        file_ = open(pulses_path +  file +  "/pulsewave.txt", "r")
        print(file)
        seq = np.fromfile(file_)

        rng = np.max(seq) - np.min(seq)
        seq = (seq - np.min(seq)) / rng - 0.5

        if 'f24_120_69_out' in file:
            seq1 = seq[144:174]
            seq1 = signal.resample(seq1, 60)
            seq2 = seq[202:232]
            seq2 = signal.resample(seq2, 60)

            abp1, abp2 = get_bp_v2(file)

            fft_ppg = np.abs(np.fft.fft(seq1))
            sample = np.zeros(feature_sz)
            sample[0:feature_sz] = abs(fft_ppg[1:feature_sz+1])

            train_sample.append(sample)
            train_y.append([abp1 / max_bp, abp2 / max_bp])

            fft_ppg = np.abs(np.fft.fft(seq2))

            sample[0:feature_sz] = abs(fft_ppg[1:feature_sz+1])
            train_sample.append(sample)
            train_y.append([abp1 / max_bp, abp2 / max_bp])

        elif 'f26_124_72_out' in file:
            abp1, abp2 = get_bp_v2(file)
            for i in range(0, 215 - 30, 30):
                seq1 = seq[i:i + 30]
                if len(seq1) == 30:
                    seq1 = signal.resample(seq1, 60)
                    fft_ppg = np.abs(np.fft.fft(seq1))
                    sample = np.zeros(feature_sz)
                    sample[0:feature_sz] = abs(fft_ppg[1:feature_sz+1])
                    train_sample.append(sample)
                    train_y.append([abp1 / max_bp, abp2 / max_bp])
            seq1 = seq[185:215]
            seq1 = signal.resample(seq1, 60)

            seq2 = seq[244:274]
            seq2 = signal.resample(seq2, 60)

            fft_ppg = np.abs(np.fft.fft(seq1))
            sample = np.zeros(feature_sz)
            sample[0:feature_sz] = abs(fft_ppg[1:feature_sz + 1])
            train_sample.append(sample)
            train_y.append([abp1 / max_bp, abp2 / max_bp])

            fft_ppg = np.abs(np.fft.fft(seq2))
            sample = np.zeros(feature_sz)
            sample[0:feature_sz] = abs(fft_ppg[1:feature_sz + 1])
            train_sample.append(sample)
            train_y.append([abp1 / max_bp, abp2 / max_bp])

        elif 'f28_110_69_out' in file:
            abp1, abp2 = get_bp_v2(file)
            for i in range(148, 148+60, 30):
                seq1 = seq[i:i + 30]
                if len(seq1) == 30:
                    seq1 = signal.resample(seq1, 60)
                    fft_ppg = np.abs(np.fft.fft(seq1))
                    sample = np.zeros(feature_sz)
                    sample[0:feature_sz] = abs(fft_ppg[1:feature_sz + 1])

                    train_sample.append(sample)
                    train_y.append([abp1 / max_bp, abp2 / max_bp])
            seq1 = seq[36:66]
            seq1 = signal.resample(seq1, 60)

            seq2 = seq[118:148]
            seq2 = signal.resample(seq2, 60)
            fft_ppg = np.abs(np.fft.fft(seq1))
            sample = np.zeros(feature_sz)
            sample[0:feature_sz] = abs(fft_ppg[1:feature_sz + 1])
            train_sample.append(sample)
            train_y.append([abp1 / max_bp, abp2 / max_bp])

            fft_ppg = np.abs(np.fft.fft(seq2))
            sample = np.zeros(feature_sz)
            sample[0:feature_sz] = abs(fft_ppg[1:feature_sz + 1])
            train_sample.append(sample)
            train_y.append([abp1 / max_bp, abp2 / max_bp])


        elif 'm48_152_103_out' in file:
            abp1, abp2 = get_bp_v2(file)
            for i in range(0, 60, 30):
                seq1 = seq[i:i + 30]
                if len(seq1) == 30:
                    seq1 = signal.resample(seq1, 60)
                    fft_ppg = np.abs(np.fft.fft(seq1))
                    sample = np.zeros(feature_sz)
                    sample[0:feature_sz] = abs(fft_ppg[1:feature_sz + 1])
                    train_sample.append(sample)
                    train_y.append([abp1 / max_bp, abp2 / max_bp])
            for i in range(165, 165 + 60, 30):
                seq1 = seq[i:i + 30]
                if len(seq1) == 30:
                    seq1 = signal.resample(seq1, 60)
                    fft_ppg = np.abs(np.fft.fft(seq1))
                    sample = np.zeros(feature_sz)
                    sample[0:feature_sz] = abs(fft_ppg[1:feature_sz + 1])
                    train_sample.append(sample)
                    train_y.append([abp1 / max_bp, abp2 / max_bp])

            else:
                abp1, abp2 = get_bp_v2(file)
                for i in range(0, len(seq),30):
                    seq1 = seq[i:i+30]
                    if len(seq1) == 30:
                        seq1 = signal.resample(seq1, 60)
                        fft_ppg = np.abs(np.fft.fft(seq1))
                        sample = np.zeros(feature_sz)
                        sample[0:feature_sz] = abs(fft_ppg[1:feature_sz + 1])
                        train_sample.append(sample)
                        train_y.append([abp1 / max_bp, abp2 / max_bp]),

    train_sample = np.float32(train_sample)
    train_y = np.float32(train_y)

    return train_sample, train_y


def get_our_v3(data_pth,train_share = 0.9):
    files = os.listdir(data_pth)
    data_set = []
    bp_set = []

    for file in files:

        if '_v2' not in file:
            file_ = open(data_pth + file + "/pulsewave.txt", "r")

            seq = []
            while True:
                seq_ = file_.readline()[:-1]
                if len(seq_) == 0:
                    break
                if seq_ != '\n':
                    seq.append(np.float32(seq_))

            print(file)
            rng = np.max(seq) - np.min(seq)
            seq = (seq - np.min(seq)) / rng - 0.5
            if 'female24_116_80' in file:

                data_set_current, bp_set_current = process_seq(seq, file, 9, 0, 60, 60, max_bp, 3)
                data_set += data_set_current
                bp_set += bp_set_current

            elif 'female24_116_88' in file:
                data_set_current, bp_set_current = process_seq(seq, file, 9, 33, 60, 60, max_bp, 3)
                data_set += data_set_current
                bp_set += bp_set_current

            elif 'female27_110_75' in file:

                data_set_current, bp_set_current = process_seq(seq, file, 9, 40, 60, 60, max_bp, 3)
                data_set += data_set_current
                bp_set += bp_set_current

            elif 'male21_113_66' in file:

                data_set_current, bp_set_current = process_seq(seq, file, 7, 54, 100, 60, max_bp, 3)
                data_set += data_set_current
                bp_set += bp_set_current

            elif 'male28_105_67' in file:

                data_set_current, bp_set_current = process_seq(seq, file, 7, 51, 100, 60, max_bp, 3)
                data_set += data_set_current
                bp_set += bp_set_current

            elif 'male30_120_75' in file:
                data_set_current, bp_set_current = process_seq(seq, file, 7, 76, 100, 60, max_bp, 3)
                data_set += data_set_current
                bp_set += bp_set_current

            elif 'male30a_124_66' in file:

                data_set_current, bp_set_current = process_seq(seq, file, 8, 20, 100, 60, max_bp, 3)
                data_set += data_set_current
                bp_set += bp_set_current



    train_x2, train_y2 = get_our_v2(data_pth + "/pulses_v2/")

    for id in range(len(train_x2)):
        data_set.append(train_x2[id])
        bp_set.append(train_y2[id])

    ids = np.arange(len(data_set), dtype=int)
    np.random.shuffle(ids)
    ids = np.array(ids)
    data_set_shuffle = []
    bp_set_shuffle = []


    for id in ids:
        data_set_shuffle.append(data_set[id])
        bp_set_shuffle.append(bp_set[id])

    data_size = len(data_set_shuffle)

    train_sample = data_set_shuffle[:int(train_share*data_size)]
    test_sample = data_set_shuffle[int(train_share*data_size):]

    train_y = bp_set_shuffle[:int(train_share * data_size)]
    test_y = bp_set_shuffle[int(train_share * data_size):]
    print("Train size " + str(train_share * data_size),"Test size " + str((1-train_share) *data_size))

    train_sample = np.float32(train_sample)
    train_y = np.float32(train_y)
    test_sample =np.float32(test_sample)
    test_y = np.float32(test_y)

    return train_sample, train_y, test_sample, test_y
if __name__ == "__main__":

    train = True
    test = True
    once = False
    model_name = 'model_1'
    train_x, train_y, test_x, test_y = get_our_v3('./pulses/', 0.8)

    if not os.path.exists("./models" + model_name):
        os.mkdir("./models/" + model_name)
    if train:
        learning_rate = 0.0001
        NN = Net()
        optimizer = optim.SGD(NN.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        batch_size = 1
        n_epoch = 500
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

            print(total_loss *180/ len(train_y))

            total_loss_[epoch] = total_loss * 180 / len(train_y)
            loss_values1 = 0
            for k in range(len(test_y)):
                data1 = torch.from_numpy(np.array(test_x[k]))
                target = torch.from_numpy(np.array(test_y[k]))
                net_out = NN(data1)
                loss1 = criterion(net_out, target)
                out = net_out.detach().numpy() * 180
                target_ = target.detach().numpy() * 180
                diff =(abs(target_[0] - out[0]) + abs(target_[1] - out[1]))/2
                loss_values1 += diff.item()

            valid_loss_[epoch] = loss_values1 / len(test_y)
            if epoch % 10 == 0:
                torch.save(NN, "./models/" + model_name + "/" + str(epoch) + ".txt")

            print('Vloss ', str(epoch), valid_loss_[epoch])
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
        torch.save(NN, "./models/" + model_name + ".txt")

    if test:
        NN = torch.load("./models/" + model_name + ".txt")
        print("validation")
        loss_values1 = 0
        for k in range(len(test_y)):
            data1 = torch.from_numpy(np.array(test_x[k]))
            net_out = NN(data1)
            print(test_y[k] *180, net_out*180)



