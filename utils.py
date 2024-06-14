import allel


def read_vcf(vcf_file):
    return allel.read_vcf(vcf_file, region=None, fields="*")
