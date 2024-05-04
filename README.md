## Installation

1. Download the query file from [here](https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/ALL.chr22.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz). After downloading, place the `vcf.gz` file in the `data` directory.
2. Install the required packages using `pip install -r requirements.txt`.
3. Install `bcftools` for VCF file manipulation. Detailed instructions can be found [here](https://samtools.github.io/bcftools/howtos/install.html).
4. Prepare dataset using `python preprocessing.py`. This will build the dataset and simulations in `data/`.
5. Run `python main_clear.py` to train and evaluate the model using clear training.
6. Run `python main_fhe.py` to train and evaluate the model using FHE.

## Methodology

We follow [gnomix](https://github.com/AI-sandbox/gnomix)'s attempt to firstly extract features from Genomic Data slices, and then smooth the features to get the final prediction. Specifically, we use the `LogisticRegression` for Genomic Data feature extraction, and `XGBClassifier` for smoothing.

The training is conducted on the `1000genomes` dataset, a large-scale Generation Genomic Dataset. After training, models will be compiled into FHE-compatible models, and then evaluated using FHE.

## Configurations

Main configurations are stored in `config.py`. Here is a brief explanation of each configuration.

- `DATA_FOLDER`, `QUERY_FILE`, `GENETIC_MAP_FILE`, `REFERENCE_FILE`, `SAMPLE_MAP_FILE`: Paths to the dataset and related files.
- `CHM`: Chromosome number. Default to 22.
- `BUILD_SPLIT`: Split the dataset into `train1` (for Phase 1), `train2` (for Phase 2) and `val` (for Evaluation) set. Default to 0.8/0.15/0.05.
- `BUILD_GENS`: Number of generations to simulate. Default to [0, 2, 4, 6, 8, 12, 16, 24, 32, 48].
- `R_ADMIXED`: Will simulate r_admixed * n_founders amount of admixed individuals.
- `SEED`: Random seed for reproducibility. Default to 42.
- `WINDOW_SIZE_CM`: Size of window in centiMorgans, use larger windows if snp density is lower. Default to 0.2.
- `SMOOTH_SIZE`: Number of windows to be taken as context for smoother. Default to 75.
- `CONTEXT_RATIO`: Context of base model windows. Default to 0.5.
- `N_JOBS`: Number of jobs to run in parallel. Default to 8.

## Hyper-parameters tuning

For better performance, some hyper-parameters, including `n_bits`, `n_estimators`, `max_depth`, can be tuned. We conducted extensive experiments on `1000genomes` dataset to find the best hyper-parameters. Note that here we will not alter any configurations for the dataset (i.e., `BUILD_GENS`, `WINDOW_SIZE_CM`, etc), as they are related to downstream tasks.

Specifically, we first tune hyper-parameters of models using `main_clear.py`. The results are shown in the following table.

| n_estimators | max_depth | Accuracy | Inference time (M1 MAX) |
|--------------|-----------|----------|-------------------------|
| 100          | 4         | 0.9811   | 0.138 s/it              |
| 50           | 4         | 0.9801   | 0.134 s/it              |
| 20           | 4         | 0.9782   | 0.133 s/it              |
| 100          | 3         | 0.9809   | 0.128 s/it              |

Then, with the above-mentioned hyper-parameters for clear training, we conducted experiments using `main_fhe.py` to determine parameters for FHE, with minimal impact on accuracy and performance. The results are shown in the following table.

| n_bits | p_error | Accuracy | Inference time (SIMULATE, M1 MAX) |
|--------|---------|----------|-----------------------------------|
| 6      | 1e-40   | 0.9811   | 2.981 s/it                        |
| 6      | 1e-2    | 0.9810   | 2.931 s/it                        |
| 6      | 1e-1    | 0.9808   | 2.935 s/it                        |
| 4      | 1e-2    | 0.9788   | 2.891 s/it                        |
| 2      | 1e-1    | 0.9636   | 2.247 s/it                        |

## Performance for reference

> Note that the following performance is tested only for reference. Depending on the hardware, downstream tasks, and set parameters, the performance may vary significantly.

Hardware: M1 MAX, 64GB RAM

Hyper-parameters:
- `n_estimators`: 100
- `max_depth`: 6
- `n_bits`: 8 for Logistic Regression, 4 for XGBClassifier
- `p_error`: 1e-2

Configurations are default as mentioned in the `config.py`. There are in total 370 base models for ensembles. According to downstream tasks, this can be adjusted by changing `WINDOW_SIZE_CM` and other related configurations.

| Dataset                 | Accuracy | Time (execute) | Time (non FHE) |
|-------------------------|----------|----------------|----------------|
| 1000genomes (Augmented) | 0.9810   | 350 min/it     | 0.164 s/it     |
