#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 23:10:57 2025

@author: willmurdock
"""

import numpy as np
import matplotlib.pyplot as plt

def make_sense_mat(NL, NR, k, s):
    # Vectorized odor vector creation
    odor_vec = np.zeros((NL, 1))
    odorant_indices = np.random.choice(NL, k, replace=False)
    odor_vec[odorant_indices] = 1

    # Vectorized binary sense matrix creation
    sense_mat = (np.random.rand(NR, NL) < s).astype(int)

    # Receptor response (binary)
    recep_vec = (sense_mat @ odor_vec != 0).astype(int)
    return sense_mat, odor_vec, recep_vec

def decode(NL, NR, k, s):
    sense_mat, odor_vec, recep_vec = make_sense_mat(NL, NR, k, s)

    # Start with all 1s and zero out odorants that are detected by inactive receptors
    odor_guess = np.ones((1, NL), dtype=int)
    inactive_receptors = np.where(recep_vec == 0)[0]
    if len(inactive_receptors) > 0:
        zero_out = np.any(sense_mat[inactive_receptors] == 1, axis=0)
        odor_guess[0, zero_out] = 0

    return odor_guess, sense_mat, odor_vec.T, recep_vec

# Parameters
NL = 10000
k = 10
s = 0.05
NR_values = np.arange(100, 600, 10)

mean_probs = []
errs = []


# Efficient simulation loop
probs=[]
for NR in NR_values:
    trial_accuracies = np.zeros(10)
    for trial in range(10):
        correct = 0
        for _ in range(100):
            odor_guess, _, odor_vec, _ = decode(NL, NR, k, s)
            correct += np.array_equal(odor_guess, odor_vec)
        trial_accuracies[trial] = correct / 100
    mean_probs.append(np.mean(trial_accuracies))
    errs.append(np.std(trial_accuracies))

# Plot
plt.errorbar(NR_values, mean_probs, yerr=errs, fmt='o', capsize=4)
plt.xlabel("Number of Receptors (NR)")
plt.ylabel("Decoding Accuracy")
plt.title("Efficient Sparse Odor Decoding vs. Receptor Count")
plt.grid(True)
plt.tight_layout()
plt.show()

'''
NR = 1000
NL = 100000
k_vals = np.arange(1,50,2)
print(len(k_vals))
s_vals = np.linspace(0.01,0.1,25)
probs=np.zeros((25,25))
sNRs = []

for i,S in enumerate(s_vals):
    print(S)
    sNRs.append(S*NR)
    for j,K in enumerate(k_vals):
        print(K)
        correct = 0
        for _ in range(5):
            odor_guess, _, odor_vec, _ = decode(NL, NR, K, S)
            correct += np.array_equal(odor_guess, odor_vec)
        probs[j][i]=(correct/100)
plt.contourf(sNRs, k_vals, probs, levels=50, cmap='viridis')
plt.colorbar(label='P(c=c)')
plt.xlabel('s*NR')
plt.ylabel('K')
plt.show()
'''

