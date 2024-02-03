# MIFAM-DTI
A drug-target interactions predicting model based on multi-source information fusion and attention mechanism
# Data
Two datasets, C.elegans and Human, were selected, and each dataset contained nodes of drugs and targets. For drugs, we extracted corresponding the physicochemical property feature vector and MACCS molecular fingerprint feature vector. For targets, we extracted the corresponding dipeptide composition feature vector and ESM-1b feature vector.
# Requirements
python = 3.7.13
torch = 1.13.1
numpy = 1.17.5
# Running
1. Merge "02protein_ESM(1).csv" and "02protein_ESM(2).csv" in the "data-celegans" and "data-human" datasets into a csv file named "02Protein_ESM.csv".
2. Adjust the file read path in utils.py.
3. Run the main.py.
