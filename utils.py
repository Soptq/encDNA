import os

import allel
import pickle

import pandas as pd


def save_dict(D, path):
    with open(path, 'wb') as handle:
        pickle.dump(D, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_dict(path):
    if not os.path.exists(path):
        return {}
    with open(path, 'rb') as handle:
        return pickle.load(handle)


def read_vcf(vcf_file, chm=None, fields=None):
    """
    Wrapper function for reading vcf files into a dictionary
    fields="*" extracts more information, take out if ruled unnecessary
    """
    if fields is None:
        fields = "*"

    data = allel.read_vcf(vcf_file, region=chm, fields=fields)

    if data is None:
        if chm is None:
            print("No data found in vcf file {}".format(vcf_file))
        else:
            print('Found no data in vcf file {} in region labeled "{}". Using all data from vcf instead...'.format(
                vcf_file, chm))
            return read_vcf(vcf_file, None, fields)

    return data


def read_genetic_map(genetic_map_path, chm=None, header=None):
    gen_map_df = pd.read_csv(genetic_map_path, delimiter="\t", header=header, comment="#", dtype=str)
    gen_map_df.columns = ["chm", "pos", "pos_cm"]

    try:
        gen_map_df = gen_map_df.astype({'chm': str, 'pos': int, 'pos_cm': float})
    except ValueError:
        if header is None:
            print("WARNING: Something wrong with genetic map format. Trying with header...")
            return read_genetic_map(genetic_map_path, chm=chm, header=0)
        else:
            raise Exception("Genetic map format not understood.")

    if chm is not None:
        chm = str(chm)
        if len(gen_map_df[gen_map_df.chm == chm]) == 0:
            gen_map_df = gen_map_df[gen_map_df.chm == "chr" + chm]
        else:
            gen_map_df = gen_map_df[gen_map_df.chm == chm]

    return gen_map_df
