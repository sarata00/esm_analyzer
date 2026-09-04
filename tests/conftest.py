import pytest

@pytest.fixture
def tensor_path():
    tensor_path = f"tests/golden_data/tensor.pth"
    return tensor_path

# Define some variables 
@pytest.fixture
def common_variables():
    amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
    mutated_sequence = "TY"
    complex_sequence = "TYVQALFDFDPQEDGELGFRRGDFIHVMDNSDPNWWKGACHGQTGMFPRNYVTPVN:VHRGPSRGSEIQPPPVNRNLKPDRKAKPTPLDL"

    return amino_acids, mutated_sequence, complex_sequence

@pytest.fixture
def common_lengths(common_variables):
    amino_acids, mutated_sequence, complex_sequence = common_variables
    n_aa = len(amino_acids)
    n_seq = len(mutated_sequence)
    n_complex = len(complex_sequence)

    return n_aa, n_seq, n_complex

@pytest.fixture
def extract_wt_index(common_variables):
    amino_acids, mutated_sequence, _ = common_variables
    first_aa = mutated_sequence[0]
    index_aa = amino_acids.index(first_aa)

    return index_aa
    