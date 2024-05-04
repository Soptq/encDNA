DATA_FOLDER = "data"

QUERY_FILE = "data/ALL.chr22.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz"
GENETIC_MAP_FILE = "data/allchrs.b37.gmap"
REFERENCE_FILE = "data/reference_1000g.vcf"
SAMPLE_MAP_FILE = "data/1000g.smap"

CHM = "22"
# how to split the data
BUILD_SPLIT = {
    "train1": 0.8,
    "train2": 0.15,
    "val": 0.05
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
