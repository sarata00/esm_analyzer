import pytest
import numpy as np

from scripts.analysis import tensors, distances

class TestDistances:

    @pytest.fixture(autouse=True)
    def setup_class(self, common_variables, common_lengths, extract_wt_index):
        tensor_path = f"tests/golden_data/tensor.pth"
        tensor = tensors.Tensor(tensor_path)
        amino_acids, mutated_sequence, _ = common_variables
        self.wt_index = extract_wt_index

        self.n_aa, self.n_seq, self.n_complex = common_lengths

        self.reshaped_tensor = tensor.tensor_reshaping(aa_list=amino_acids, mutated_sequence=mutated_sequence)
        self.dim = self.reshaped_tensor.shape[-1]

        self.dis = distances.Distances(tensor=self.reshaped_tensor, mutated_sequence=mutated_sequence)

    def test_find_wt(self, extract_wt_index):
        wt = self.dis.find_wt()

        test_wt = self.reshaped_tensor[0, self.wt_index, :, :]

        assert wt.shape == (self.n_complex + 2, self.dim), "Wildtype tensor has different shape than expected."
        assert np.array_equal(wt, test_wt), "Wildtype tensor is not well generated."

    def test_calculate_differences(self):
        differences = self.dis.calculate_diferences()
        wt = self.reshaped_tensor[0, self.wt_index, :, :]
        tensor_differences = self.reshaped_tensor - wt

        assert differences.shape == (self.n_seq, self.n_aa, self.n_complex + 2, self.dim), "Tensor of differences has different shape than expected."
        assert np.array_equal(differences, tensor_differences), "Tensor of differences is not well generated."

    def test_select_receptor_residues(self):
        receptor_tensor = self.dis.select_receptor_residues()
        assert receptor_tensor.shape == (self.n_seq, self.n_aa, self.n_seq, self.dim), "Receptor residues selection failed."

    def test_remove_special_tokens(self):
        tensor = self.dis.remove_special_tokens()
        assert tensor.shape == (self.n_seq, self.n_aa, self.n_complex - 1, self.dim), "Special tokens removal failed."

    def test_get_euclidean_distance(self):
        pass

