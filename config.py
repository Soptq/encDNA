DATA_FOLDER = "data"

# 1000 Genomes
QUERY_FILE = "data/ALL.chr22.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz"
GENETIC_MAP_FILE = "data/allchrs.b37.gmap"
REFERENCE_FILE = "data/reference_1000g.vcf"
SAMPLE_MAP_FILE = "data/1000g.smap"

# https://github.com/alephzerox/ancestry-fhe
REFERENCE_PANEL_SAMPLES = "data/reference_panel_samples.vcf.gz"
REFERENCE_PANEL_MAPPING = "data/reference_panel_mapping.csv"
TEST_SAMPLES = "data/test_samples.vcf.gz"
TEST_MAPPING = "data/test_mapping.pickle.gz"

CHM = "22"
# how to split the data
BUILD_SPLIT = {
    "train1": 0.8,
    "train2": 0.2,
}
NUM_SPLIT = {
    "train1": 2400,
    "train2": 600,
}
# which generation to simulate, not critical some accuracy can be squeezed if it better represents the query data
BUILD_GENS = [0, 2, 4, 6, 8, 12, 16, 24, 32, 48]
# we simulate r_admixed*n_founders amount of admixed individuals
R_ADMIXED = 1
SEED = 42

# training related
WINDOW_SIZE_CM = 0.2
SMOOTH_SIZE = 75
CONTEXT_RATIO = 0.5
N_JOBS = 8
