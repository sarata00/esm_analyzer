#!/usr/bin/env python3
import esm
import torch
import numpy as np
import pandas as pd
import argparse
import time
import os

# Define the amino acids
amino_acids = list("ACDEFGHIKLMNPQRSTVWY")

def read_fasta_sequence(fasta_path):
    fasta_file = open(fasta_path, "r")
    seq = []
    for line in fasta_file:
        if not line.startswith(">"):
            seq.append("".join(line.split()))
    protein_sequence = "".join(seq)

    fasta_file.close()
    
    return protein_sequence


def load_esm_model(model_name):
    """Load ESM-2 model"""
    models = {
        "esm2_t6_8M_UR50D": esm.pretrained.esm2_t6_8M_UR50D,
        "esm2_t12_35M_UR50D": esm.pretrained.esm2_t12_35M_UR50D,
        "esm2_t30_150M_UR50D": esm.pretrained.esm2_t30_150M_UR50D,
        "esm2_t33_650M_UR50D": esm.pretrained.esm2_t33_650M_UR50D,
        "esm2_t36_3B_UR50D": esm.pretrained.esm2_t36_3B_UR50D,
        "esm2_t48_15B_UR50D": esm.pretrained.esm2_t48_15B_UR50D,
    }
    
    model, alphabet = models[model_name]()
    n_layers = model.num_layers

    return model, alphabet, n_layers


def calculate_llr(model_name, protein_sequence, amino_acids, start_pos=1, end_pos=None):
    
    # Load the model and tokenizer
    model, alphabet, _ = load_esm_model(model_name=model_name)
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results

    data = [("input_protein", protein_sequence)]

    # Batch labels: name of the protein
    # Batch_strs: protein sequence
    # Batch_tokens: sequence tokens
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    # Tokenize the input sequence
    sequence_length = batch_tokens.shape[1] - 2 # Excluding the special tokens

    # Adjust end position if not specified
    if end_pos is None:
        end_pos = sequence_length

    # Initialize heatmap
    heatmap = np.zeros((20, end_pos - start_pos + 1)) # y axis = 20 amino acids
                                                      # x axis = sequence length
    # Calculate LLRs for each position and amino acid
    for position in range(start_pos, end_pos + 1):

        # Mask the target position
        masked_input_ids = batch_tokens.clone()
        # For each position, we are masking the token in the given position
        masked_input_ids[0, position] = alphabet.mask_idx

        # Get logits of all tokens when the current position token is masked
        with torch.no_grad():
            results = model(masked_input_ids)
            logits = results["logits"]  # logits.shape = torch.size([1,58,33])

        # Calculate log probabilities
            # Here we are taking the logits for the masked position: logits[0,position] (shape = torch.size([33]))
            # softmax helps us to normalize the logits
        probabilities = torch.nn.functional.softmax(logits[0, position], dim=0) # softmax helps us to normalize the logits
        log_probabilities = torch.log(probabilities)

        # Get the log probability of the wild-type residue
        wt_residue = batch_tokens[0, position].item()      # token value for the current position token (for example, position 2 the corresponding token is 19, so wt_residue=19)
        # From the all the probabilities of each amino acid in the vocabulary
        # for that position, only extract the one corresponding to the wt amino acid
        log_prob_wt = log_probabilities[wt_residue].item()

        for i, amino_acid in enumerate(amino_acids):
            # Now extract the probabilities for the rest of amino acids, which are stored in a list
            log_prob_mt = log_probabilities[alphabet.tok_to_idx[amino_acid]].item()
            # Calculate the difference between the current variant and the wiltype probability
            # For this given position
            difference = log_prob_mt-log_prob_wt

            # Add for a given mutant amino acid and position the difference
            # between that variant and wildtype protein
            heatmap[i, position-start_pos] = difference

    # Store the results in a dataframe
    position_index = range(start_pos, end_pos + 1)
    residue_index = [protein_sequence[i - 1] for i in position_index]
    df = pd.DataFrame(heatmap, index=amino_acids, columns=residue_index)

    return df


def main():
    # 1. Define input and output arguments
    parser= argparse.ArgumentParser(description="This script calculates the log-likelihood ratio to evaluate the effect of single mutations in a given protein sequence. This code is an adaptation from AmelieSchreiber code (link: https://huggingface.co/blog/AmelieSchreiber/mutation-scoring)",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-i", "--input_sequence",type=str, help="input fasta file")
    parser.add_argument("-m", "--model", type=str ,help=" specify the ESM-2 model you want to use. ")
    parser.add_argument("-o", "--output_path",type=str, help="output path.",default="./")
    args = vars(parser.parse_args())

    # Validate the input arguments
    if not os.path.exists(args["input_sequence"]):
        raise FileNotFoundError(f"Error, the input fasta file {args['input_sequence']} does not exist.")
    if args["model"] not in ["esm2_t6_8M_UR50D", "esm2_t12_35M_UR50D", "esm2_t30_150M_UR50D", "esm2_t33_650M_UR50D", "esm2_t36_3B_UR50D", "esm2_t48_15B_UR50D"]:
        raise ValueError(f"Error, the model {args['model']} is not available.")
    if not os.path.exists(args["output_path"]):
        raise FileNotFoundError(f"Error, the output path {args['output_path']} does not exist.")
    

    # 2. Read the FASTA file and arguments
    protein_sequence = read_fasta_sequence(fasta_path=args["input_sequence"])     
    output_path = args["output_path"]        
    checkpoint = args["model"]       # model name

    # 3. Run the LLR calculation
    start = time.time() 
    df = calculate_llr(model_name=checkpoint, 
                       protein_sequence=protein_sequence, 
                       amino_acids=amino_acids)
    end = time.time()          
    print(f"It takes {end - start} seconds")

    # 4. Create two types of DataFrames: df_heatmap and df_llr_result
    df_llr_result = df.reset_index().melt(id_vars="index", var_name="WT", value_name="LLR")
    df_llr_result.rename(columns={"index": "Mut"}, inplace=True)

    df_pos=[]
    for i, aa1 in enumerate(protein_sequence):
        for j, aa2 in enumerate(amino_acids):
            df_pos.append(i+1)
    df_llr_result["Pos"] = df_pos   # add a position column


    # 5. Save the results
    output_dir = os.path.abspath(output_path)                # Get the absolute path of the output directory
    if not os.path.exists(f"{output_dir}/{checkpoint}"):     # Check if the parent directory exists
        os.makedirs(f"{output_dir}/{checkpoint}")            # If not, create it
    
    df_path_heatmap = os.path.join(f"{output_dir}/{checkpoint}", f"HEATMAP_LLR_results_{checkpoint}.csv")
    df_path_llr_results = os.path.join(f"{output_dir}/{checkpoint}", f"LLR_results_{checkpoint}.csv")
    
    df.to_csv(df_path_heatmap) # save the heatmap table
    df_llr_result.to_csv(df_path_llr_results)


if __name__ == "__main__":
    main()