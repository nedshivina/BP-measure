import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import os

feature_sz = 50
max_bp = 180


def process_seq(seq, abp, ppg_start, ppg_len, ppg_gt_len, bp_max):
    data_set = []
    bp_set = []


    for i in range(ppg_start, len(seq), ppg_len):
        seq1 = seq[i:i + ppg_len]
        seq1 = signal.resample(seq1, ppg_gt_len)
        fft_ppg = np.abs(np.fft.fft(seq1))
        if len(seq1) == ppg_gt_len:
            sample = np.zeros(feature_sz)
            sample[0:feature_sz] = abs(fft_ppg[1:feature_sz + 1])
            data_set.append(sample)
            bp_set.append([abp[0] / bp_max, abp[1] / bp_max])
    return data_set, bp_set


def get_bp_v2(filename):


    bp_str = filename[4:-3]
    abp1 = np.float32(bp_str[:3])
    abp2 = np.float32(bp_str[4:])

    return abp1, abp2
def get_feature_set_v2(seq, pulse_name, abp):


    sample_x = []
    sample_y = []


    if 'f24_120_69_v2' in pulse_name:
        seq1 = seq[144:174]
        seq1 = signal.resample(seq1, 60)
        seq2 = seq[202:232]
        seq2 = signal.resample(seq2, 60)

        fft_ppg = np.abs(np.fft.fft(seq1))
        sample = np.zeros(feature_sz)
        sample[0:feature_sz] = abs(fft_ppg[1:feature_sz+1])

        sample_x.append(sample)
        sample_y.append([abp[0] / max_bp, abp[1] / max_bp])

        fft_ppg = np.abs(np.fft.fft(seq2))

        sample[0:feature_sz] = abs(fft_ppg[1:feature_sz+1])
        sample_x.append(sample)
        sample_y.append([abp[0] / max_bp, abp[1] / max_bp])

    elif 'f26_124_72_v2' in pulse_name:

        for i in range(0, 215 - 30, 30):
            seq1 = seq[i:i + 30]
            if len(seq1) == 30:
                seq1 = signal.resample(seq1, 60)
                fft_ppg = np.abs(np.fft.fft(seq1))
                sample = np.zeros(feature_sz)
                sample[0:feature_sz] = abs(fft_ppg[1:feature_sz+1])
                sample_x.append(sample)
                sample_y.append([abp[0] / max_bp, abp[1] / max_bp])
        seq1 = seq[185:215]
        seq1 = signal.resample(seq1, 60)

        seq2 = seq[244:274]
        seq2 = signal.resample(seq2, 60)

        fft_ppg = np.abs(np.fft.fft(seq1))
        sample = np.zeros(feature_sz)
        sample[0:feature_sz] = abs(fft_ppg[1:feature_sz + 1])
        sample_x.append(sample)
        sample_y.append([abp[0] / max_bp, abp[1] / max_bp])

        fft_ppg = np.abs(np.fft.fft(seq2))
        sample = np.zeros(feature_sz)
        sample[0:feature_sz] = abs(fft_ppg[1:feature_sz + 1])
        sample_x.append(sample)
        sample_y.append([abp[0] / max_bp, abp[1] / max_bp])

    elif 'f28_110_69_v2' in pulse_name:

        for i in range(148, 148+60, 30):
            seq1 = seq[i:i + 30]
            if len(seq1) == 30:
                seq1 = signal.resample(seq1, 60)
                fft_ppg = np.abs(np.fft.fft(seq1))
                sample = np.zeros(feature_sz)
                sample[0:feature_sz] = abs(fft_ppg[1:feature_sz + 1])

                sample_x.append(sample)
                sample_y.append([abp[0] / max_bp, abp[1] / max_bp])
        seq1 = seq[36:66]
        seq1 = signal.resample(seq1, 60)

        seq2 = seq[118:148]
        seq2 = signal.resample(seq2, 60)
        fft_ppg = np.abs(np.fft.fft(seq1))
        sample = np.zeros(feature_sz)
        sample[0:feature_sz] = abs(fft_ppg[1:feature_sz + 1])
        sample_x.append(sample)
        sample_y.append([abp[0] / max_bp, abp[1] / max_bp])

        fft_ppg = np.abs(np.fft.fft(seq2))
        sample = np.zeros(feature_sz)
        sample[0:feature_sz] = abs(fft_ppg[1:feature_sz + 1])
        sample_x.append(sample)
        sample_y.append([abp[0] / max_bp, abp[1] / max_bp])


    elif 'm48_152_103_v2' in pulse_name:
        for i in range(0, 60, 30):
            seq1 = seq[i:i + 30]
            if len(seq1) == 30:
                seq1 = signal.resample(seq1, 60)
                fft_ppg = np.abs(np.fft.fft(seq1))
                sample = np.zeros(feature_sz)
                sample[0:feature_sz] = abs(fft_ppg[1:feature_sz + 1])
                sample_x.append(sample)
                sample_y.append([abp[0] / max_bp, abp[1] / max_bp])
        for i in range(165, 165 + 60, 30):
            seq1 = seq[i:i + 30]
            if len(seq1) == 30:
                seq1 = signal.resample(seq1, 60)
                fft_ppg = np.abs(np.fft.fft(seq1))
                sample = np.zeros(feature_sz)
                sample[0:feature_sz] = abs(fft_ppg[1:feature_sz + 1])
                sample_x.append(sample)
                sample_y.append([abp[0] / max_bp, abp[1] / max_bp])

        else:
            for i in range(0, len(seq),30):
                seq1 = seq[i:i+30]
                if len(seq1) == 30:
                    seq1 = signal.resample(seq1, 60)
                    fft_ppg = np.abs(np.fft.fft(seq1))
                    sample = np.zeros(feature_sz)
                    sample[0:feature_sz] = abs(fft_ppg[1:feature_sz + 1])
                    sample_x.append(sample)
                    sample_y.append([abp[0] / max_bp, abp[1] / max_bp])

    return sample_x, sample_y


def get_feature_set(data_pth, show_flag=False):
    files = os.listdir(data_pth)
    data_set = []
    bp_set = []

    for file in files:
        file_ = open(data_pth + file + "/pulsewave.txt", "r")
        seq = []
        while True:
            seq_ = file_.readline()[:-1]
            if len(seq_) == 0:
                break
            if seq_ != '\n':
                seq.append(np.float32(seq_))

        file_.close()
        file_ = open(data_pth + file + "/pressure.txt", "r")
        abp = np.fromfile(file_, dtype=int, sep="\t", count=2)


        print(file)
        rng = np.max(seq) - np.min(seq)
        seq = (seq - np.min(seq)) / rng - 0.5
        if show_flag:
            plt.clf()
            plt.plot(seq[:300], label=file)
            plt.legend()
            plt.show()
        if '_v3' in file:

            if 'f24_116_80' in file:

                data_set_current, bp_set_current = process_seq(seq, abp, 0, 60, 60, max_bp)
                data_set += data_set_current
                bp_set += bp_set_current

            elif 'f24_116_88' in file:
                data_set_current, bp_set_current = process_seq(seq, abp, 33, 60, 60, max_bp)
                data_set += data_set_current
                bp_set += bp_set_current

            elif 'f27_110_75' in file:
                data_set_current, bp_set_current = process_seq(seq, abp, 40, 60, 60, max_bp)
                data_set += data_set_current
                bp_set += bp_set_current

            elif 'm21_113_66' in file:
                data_set_current, bp_set_current = process_seq(seq, abp, 54, 100, 60, max_bp)
                data_set += data_set_current
                bp_set += bp_set_current

            elif 'm28_105_67' in file:

                data_set_current, bp_set_current = process_seq(seq, abp, 51, 100, 60, max_bp)
                data_set += data_set_current
                bp_set += bp_set_current

            elif 'm30_120_75' in file:
                data_set_current, bp_set_current = process_seq(seq, abp, 76, 100, 60, max_bp)
                data_set += data_set_current
                bp_set += bp_set_current

            elif 'm30a_124_66' in file:

                data_set_current, bp_set_current = process_seq(seq, abp, 20, 100, 60, max_bp)
                data_set += data_set_current
                bp_set += bp_set_current
        elif "_v2" in file:
            data_set_current, bp_set_current = get_feature_set_v2(seq, file, abp)
            data_set += data_set_current
            bp_set += bp_set_current
        else:  #any other data
            data_set_current, bp_set_current = process_seq(seq, abp, 0, 60, 60, max_bp)
            data_set += data_set_current
            bp_set += bp_set_current



    data_set = np.array(data_set)
    bp_set = np.array(bp_set)

    return data_set, bp_set


def save_feature_set(save_path, feature_x, feature_y):


    np.save(save_path + "/feature_set_x.npy", feature_x)
    np.save(save_path + "/feature_set_y.npy", feature_y)
    print(str(len(feature_x)) + " features was prepared and saved to " + save_path)

    return None

import sys
if __name__ == "__main__":


    args = sys.argv
    feature_x, feature_y = get_feature_set(args[1], False)
    save_feature_set(args[2], feature_x, feature_y)



