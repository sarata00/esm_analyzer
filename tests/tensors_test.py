from scripts.analysis import tensors

def test_load_tensor(tensor_path):
    tensor = tensors.Tensor(tensor_path)

    assert tensor.tensor_file is not None, "Tensor file is not loaded."

def test_tensor_reshaping(tensor_path, common_variables):
    t = tensors.Tensor(tensor_path)
    # Load common variables
    amino_acids, mutated_sequence, complex_sequence = common_variables
    n_aa = len(amino_acids)
    n_seq = len(mutated_sequence)
    n_complex = len(complex_sequence)
    dim = t.tensor_file.shape[-1]

    reshaped_tensor = t.tensor_reshaping(aa_list=amino_acids, mutated_sequence=mutated_sequence)

    assert reshaped_tensor.shape == (n_seq, n_aa, n_complex + 2, dim), "Tensor reshaping failed."