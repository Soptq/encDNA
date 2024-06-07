import os

import pandas as pd
import numpy as np

from scipy import stats

from config import *
from utils import save_dict, read_genetic_map, load_dict
from datasets import LAIDataset, AncestryDataset


def build_dataset():
    lai_dataset = LAIDataset(CHM, REFERENCE_FILE, GENETIC_MAP_FILE, seed=SEED)
    lai_dataset.build_dataset(SAMPLE_MAP_FILE)

    sample_map_path = os.path.join(DATA_FOLDER, "sample_maps")
    if not os.path.exists(sample_map_path):
        os.makedirs(sample_map_path)

    # split sample map and write it.
    lai_dataset.create_splits(BUILD_SPLIT, sample_map_path)

    # write metadata into data_path/metadata.yaml
    save_dict(lai_dataset.metadata(), os.path.join(DATA_FOLDER, "metadata.pkl"))
    # Save genetic map df and store it inside model later after training
    gen_map_df = read_genetic_map(GENETIC_MAP_FILE, CHM)
    save_dict(gen_map_df, os.path.join(DATA_FOLDER, "gen_map_df.pkl"))

    # get num_outs
    num_outs = {}
    min_splits = {"train1": 800, "train2": 150, "val": 50}
    for split in BUILD_SPLIT:
        total_sim = max(len(lai_dataset.return_split(split)) * R_ADMIXED, min_splits[split])
        num_outs[split] = int(total_sim / len(BUILD_GENS))

    print("Running Simulation...")
    for split in BUILD_SPLIT:
        split_path = os.path.join(DATA_FOLDER, split)
        if not os.path.exists(split_path):
            os.makedirs(split_path)
        for gen in BUILD_GENS:
            lai_dataset.simulate(num_outs[split],
                                 split=split,
                                 gen=gen,
                                 outdir=os.path.join(split_path, "gen_" + str(gen)),
                                 return_out=False)

    return


def preprocess():
    """
    Preprocess the data to get the reference file from the provided query file.

    This step is a part of the data preprocessing pipeline, so it is done in a clear way, without FHE.
    :return:
    """
    sample_map = pd.read_csv(SAMPLE_MAP_FILE, sep="\t")
    samples = list(sample_map["#Sample"])
    np.savetxt("data/samples_1000g.tsv", samples, delimiter="\t", fmt="%s")
    subset_cmd = "bcftools view" + " -S data/samples_1000g.tsv -o " + REFERENCE_FILE + " " + QUERY_FILE
    print("Running in command line: \n\t", subset_cmd)
    os.system(subset_cmd)
    build_dataset()


def build_dataset_v2():
    sample_map_path = os.path.join(DATA_FOLDER, "sample_maps")
    if not os.path.exists(sample_map_path):
        os.makedirs(sample_map_path)

    reference_panel = AncestryDataset(REFERENCE_PANEL_SAMPLES, REFERENCE_PANEL_MAPPING, R_ADMIXED, BUILD_SPLIT, BUILD_GENS, TEST_SAMPLES, TEST_MAPPING, DATA_FOLDER)
    save_dict(reference_panel.metadata(), os.path.join(DATA_FOLDER, "metadata.pkl"))


def preprocess_v2():
    build_dataset_v2()


def load_np_data(files):
    data = []
    for f in files:
        data.append(np.load(f).astype(np.int16))
    data = np.concatenate(data, axis=0)
    return data


def dropout_row(data, missing_percent):
    num_drops = int(len(data) * missing_percent)
    drop_indices = np.random.choice(np.arange(len(data)), size=num_drops, replace=False)
    data[drop_indices] = 2
    return data


def simulate_missing_values(data, missing_percent=0.0):
    if missing_percent == 0:
        return data
    return np.apply_along_axis(dropout_row, axis=1, arr=data, missing_percent=missing_percent)


def window_reshape(data, win_size):
    """
    Takes in data of shape (N, chm_len), aggregates labels and
    returns window shaped data of shape (N, chm_len//window_size)
    """

    # Split in windows and make the last one contain the remainder
    chm_len = data.shape[1]
    drop_last_idx = chm_len // win_size * win_size - win_size
    window_data = data[:, 0:drop_last_idx]
    rem = data[:, drop_last_idx:]

    # reshape accordingly
    N, C = window_data.shape
    num_winds = C // win_size
    window_data = window_data.reshape(N, num_winds, win_size)

    # attach thet remainder
    window_data = stats.mode(window_data, axis=2)[0].squeeze()
    rem_label = stats.mode(rem, axis=1)[0].squeeze()
    window_data = np.concatenate((window_data, rem_label[:, np.newaxis]), axis=1)

    return window_data


def data_process(X, labels, window_size, missing=0.0):
    """
    Takes in 2 numpy arrays:
        - X is of shape (N, chm_len)
        - labels is of shape (N, chm_len)

    And returns 2 processed numpy arrays:
        - X is of shape (N, chm_len)
        - labels is of shape (N, chm_len//window_size)
    """

    # Reshape labels into windows
    y = window_reshape(labels, window_size)

    # simulates lacking of input
    if missing != 0:
        print("Simulating missing values...")
        X = simulate_missing_values(X, missing)

    X = np.array(X, dtype="int8")
    y = np.array(y, dtype="int16")

    return X, y


def get_data():
    metadata = load_dict(os.path.join(DATA_FOLDER, "metadata.pkl"))
    snp_pos = metadata["pos_snps"]
    snp_ref = metadata["ref_snps"]
    snp_alt = metadata["alt_snps"]
    pop_order = metadata["num_to_pop"]
    pop_list = []
    for i in range(len(pop_order.keys())):
        pop_list.append(pop_order[i])
    pop_order = np.array(pop_list)

    A = len(pop_order)
    C = len(snp_pos)
    M = int(round(WINDOW_SIZE_CM * (C / (100 * metadata["morgans"]))))
    if C % M == 0:
        M -= 1

    meta = {
        "A": A,  # number of ancestry
        "C": C,  # chm length
        "M": M,  # window size in SNPs
        "snp_pos": snp_pos,
        "snp_ref": snp_ref,
        "snp_alt": snp_alt,
        "pop_order": pop_order
    }

    def read(split):
        paths = [os.path.join(DATA_FOLDER, split, "gen_" + str(gen)) for gen in BUILD_GENS]
        X_files = [p + "/mat_vcf_2d.npy" for p in paths]
        labels_files = [p + "/mat_map.npy" for p in paths]
        X_raw, labels_raw = [load_np_data(f) for f in [X_files, labels_files]]
        X, y = data_process(X_raw, labels_raw, M)
        return X, y

    X_t1, y_t1 = read("train1")
    X_t2, y_t2 = read("train2")
    X_v, y_v = read("val")

    X_t1 = X_t1.astype(np.float32)
    y_t1 = y_t1.astype(np.int32)
    X_t2 = X_t2.astype(np.float32)
    y_t2 = y_t2.astype(np.int32)
    X_v = X_v.astype(np.float32)
    y_v = y_v.astype(np.int32)

    data = ((X_t1, y_t1), (X_t2, y_t2), (X_v, y_v))

    return data, meta


def get_data_v2():
    metadata = load_dict(os.path.join(DATA_FOLDER, "metadata.pkl"))
    snp_pos = metadata["pos_snps"]
    snp_ref = metadata["ref_snps"]
    snp_alt = metadata["alt_snps"]
    pop_order = metadata["num_to_pop"]

    pop_order = {int(k): v for k, v in pop_order.items()}

    pop_list = []
    for i in range(len(pop_order.keys())):
        pop_list.append(pop_order[i])
    pop_order = np.array(pop_list)

    A = len(pop_order)
    C = len(snp_pos)
    M = 2048
    if C % M == 0:
        M -= 1

    meta = {
        "A": A,  # number of ancestry
        "C": C,  # chm length
        "M": M,  # window size in SNPs
        "snp_pos": snp_pos,
        "snp_ref": snp_ref,
        "snp_alt": snp_alt,
        "pop_order": pop_order
    }

    def read(split):
        if split == "val":
            sample_file_path = os.path.join(DATA_FOLDER, split, "gen_0/mat_vcf_2d.npy")
            mapping_file_path = os.path.join(DATA_FOLDER, split, "gen_0/mat_map.npy")
            X_raw = load_np_data([sample_file_path])
            labels_raw = load_np_data([mapping_file_path])
        else:
            paths = [os.path.join(DATA_FOLDER, split, "gen_" + str(gen)) for gen in BUILD_GENS]
            X_files = [p + "/mat_vcf_2d.npy" for p in paths]
            labels_files = [p + "/mat_map.npy" for p in paths]
            X_raw, labels_raw = [load_np_data(f) for f in [X_files, labels_files]]
        X, y = data_process(X_raw, labels_raw, M)
        return X, y

    X_t1, y_t1 = read("train1")
    X_t2, y_t2 = read("train2")
    X_v, y_v = read("val")

    X_t1 = X_t1.astype(np.float32)
    y_t1 = y_t1.astype(np.int32)
    X_t2 = X_t2.astype(np.float32)
    y_t2 = y_t2.astype(np.int32)
    X_v = X_v.astype(np.float32)
    y_v = y_v.astype(np.int32)

    data = ((X_t1, y_t1), (X_t2, y_t2), (X_v, y_v))

    return data, meta


if __name__ == '__main__':
    # preprocess()
    preprocess_v2()
