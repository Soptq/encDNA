import copy
import os
import random

import pandas as pd
import numpy as np

from scipy import stats

from config import *

from utils import read_vcf
import pickle
import gzip


def preprocess():
    print("Reading VCF file...")
    reference_vcf_file = read_vcf(REFERENCE_PANEL_SAMPLES)
    test_vcf_file = read_vcf(TEST_SAMPLES)

    call_data = reference_vcf_file["calldata/GT"]
    vcf_samples = reference_vcf_file["samples"]

    test_call_data = test_vcf_file["calldata/GT"]
    test_vcf_samples = test_vcf_file["samples"]

    pos_snps = reference_vcf_file["variants/POS"]
    num_snps = call_data.shape[0]

    print("Getting sample map info...")
    reference_mapping = pd.read_csv(REFERENCE_PANEL_MAPPING, dtype="object")
    reference_mapping.columns = ["sample", "population", "population_code"]
    pop_to_num = {}
    for i, pop in enumerate(reference_mapping["population"]):
        pop_to_num[pop] = reference_mapping["population_code"][i]
    reference_mapping["population_code"] = reference_mapping["population"].apply(pop_to_num.get)
    num_to_pop = {v: k for k, v in pop_to_num.items()}

    pop_order = {int(k): v for k, v in num_to_pop.items()}
    pop_list = []
    for i in range(len(pop_order.keys())):
        pop_list.append(pop_order[i])
    pop_order = np.array(pop_list)
    A = len(pop_order)
    C = len(pos_snps)

    metadata = {
        "A": A,
        "C": C,
    }
    with open(os.path.join(DATA_FOLDER, "metadata.pkl"), 'wb') as handle:
        pickle.dump(metadata, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with gzip.open(TEST_MAPPING, "rb") as mapping_file:
        mappings = pickle.load(mapping_file)
    test_sample_map = {}
    for name, anc_1, anc_2 in mappings:
        test_sample_map[name] = [anc_1, anc_2]

    print("Reading founders...")
    founders = {}
    for i in reference_mapping.iterrows():
        sample_name = i[1]["sample"]
        index = vcf_samples.tolist().index(sample_name)

        snp_1 = call_data[:, index, 0].astype(int)
        snp_2 = call_data[:, index, 1].astype(int)
        label = i[1]["population_code"]

        if label not in founders:
            founders[label] = [(snp_1, snp_2, label)]
        else:
            founders[label].append((snp_1, snp_2, label))

    splitted_founders = {k: [] for k in BUILD_SPLIT.keys()}
    for l, f in founders.items():
        n_founders = len(f)
        f_idx = np.arange(n_founders)
        np.random.shuffle(f_idx)
        build_split = np.array(list(BUILD_SPLIT.values())) * n_founders
        cumsum_split = np.cumsum(build_split).astype(int)
        splitted = np.split(f_idx, cumsum_split.tolist())
        for i, k in enumerate(BUILD_SPLIT.keys()):
            split_idx = splitted[i]
            for _idx in split_idx:
                snp_1, snp_2, label = f[_idx]
                label_1 = label_2 = np.array([label] * num_snps, dtype=int)
                splitted_founders[k].append((snp_1, snp_2, label_1, label_2))

    test_founders = []
    test_samples = test_call_data.transpose(1, 2, 0)
    for i in range(len(test_samples)):
        snp_1 = test_samples[i, 0, :].astype(int)
        snp_2 = test_samples[i, 1, :].astype(int)
        label_1 = test_sample_map[test_vcf_samples[i]][0].astype(int)
        label_2 = test_sample_map[test_vcf_samples[i]][1].astype(int)
        test_founders.append([snp_1, snp_2, label_1, label_2])
    np.save(os.path.join(DATA_FOLDER, f"test.npy"), test_founders)

    print("Simulating...")
    num_sims = {k: int(NUM_SPLIT[k] / len(BUILD_GENS)) for k in BUILD_SPLIT.keys()}
    for splitted_key in BUILD_SPLIT.keys():
        admix = []
        for gen in BUILD_GENS:
            print(f"Simulating for {splitted_key}-gen{gen}")
            if gen == 0:
                continue
            for i in range(num_sims[splitted_key]):
                admix.append([None, None, None, None])
                for j in range(2):
                    founder_id = np.random.choice(len(splitted_founders[splitted_key]))
                    snp_1, snp_2, label_1, label_2 = copy.deepcopy(splitted_founders[splitted_key][founder_id])
                    snp, label = (snp_1, label_1) if random.random() < 0.5 else (snp_2, label_2)
                    breakpoints = np.random.choice(range(1, num_snps), size=int(sum(np.random.poisson(0.75, size=gen))) + 1, replace=False)
                    breakpoints = np.concatenate(([0], np.sort(breakpoints), [num_snps]))
                    for i in range(len(breakpoints) - 1):
                        _founder_id = np.random.choice(len(splitted_founders[splitted_key]))
                        _snp_1, _snp_2, _label_1, _label_2 = copy.deepcopy(splitted_founders[splitted_key][_founder_id])
                        _snp, _label = (_snp_1, _label_1) if random.random() < 0.5 else (_snp_2, _label_2)
                        print(
                            f"Founders: {founder_id}:{label} -> {_founder_id}:{_label}, {breakpoints[i]}:{breakpoints[i + 1]}")
                        snp[breakpoints[i]:breakpoints[i + 1]] = _snp[breakpoints[i]:breakpoints[i + 1]].copy()
                        label[breakpoints[i]:breakpoints[i + 1]] = _label[breakpoints[i]:breakpoints[i + 1]].copy()
                    admix[-1][j], admix[-1][j + 2] = snp, label
        for v in splitted_founders[splitted_key]:
            admix.append(v)
        np.save(os.path.join(DATA_FOLDER, f"{splitted_key}.npy"), admix)


def slide_window(data, smooth_win_size, y=None):
    N, W, A = data.shape

    pad = (smooth_win_size + 1) // 2
    data_padded = np.pad(data, ((0, 0), (pad, pad), (0, 0)), mode='reflect')
    X_slide = np.lib.stride_tricks.sliding_window_view(data_padded, (1, smooth_win_size, A))
    X_slide = X_slide[:, :W, :].reshape(N * W, -1)
    y_slide = None if y is None else y.reshape(N * W)

    return X_slide, y_slide


def get_data():
    with open(os.path.join(DATA_FOLDER, "metadata.pkl"), 'rb') as f:
        meta = pickle.load(f)
    M = 2048
    M = M if meta["C"] % M == 0 else M - 1
    meta["M"] = M

    def read(split, shuffle=True):
        file_path = os.path.join(DATA_FOLDER, split + ".npy")
        with open(file_path, 'rb') as f:
            data = np.load(f, allow_pickle=True)
        if shuffle:
            np.random.shuffle(data)
        X, y = [], []
        for d in data:
            snp_1, snp_2, label_1, label_2 = d
            X.append(snp_1)
            X.append(snp_2)
            y.append(label_1)
            y.append(label_2)

        X = np.array(X)
        y = np.array(y)

        N, L = y.shape
        y = y[:, 0:L // M * M].reshape(N, L // M, M)
        y = stats.mode(y, axis=2)[0].squeeze()

        y = np.swapaxes(y, 0, 1)
        n_classes = len(np.unique(y))
        for i in range(len(y)):
            if np.unique(y[i]).shape[0] != n_classes:
                missed_classes = np.setdiff1d(np.arange(n_classes), y[i])
                for missed_class in missed_classes:
                    random_idx = np.random.choice(y.shape[1], 2)
                    y[i, random_idx] = missed_class
        for i in range(len(y)):
            assert np.unique(y[i]).shape[0] == n_classes
        y = np.swapaxes(y, 0, 1)

        return X, y

    X_t1, y_t1 = read("train1")
    X_t2, y_t2 = read("train2")
    X_t, y_t = read("test", shuffle=False)

    X_t1 = X_t1.astype(np.float32)
    y_t1 = y_t1.astype(np.int32)
    X_t2 = X_t2.astype(np.float32)
    y_t2 = y_t2.astype(np.int32)
    X_t = X_t.astype(np.float32)
    y_t = y_t.astype(np.int32)

    data = ((X_t1, y_t1), (X_t2, y_t2), (X_t, y_t))

    return data, meta


if __name__ == '__main__':
    preprocess()
