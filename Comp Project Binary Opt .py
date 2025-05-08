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

def estpc(Nr,k,s):
    alpha = k/NL
    return (alpha+(1-alpha)*(1-(1-s*(1-s*alpha)**(NL-1))**Nr))**NL

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
    print(NR)
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
plt.plot(NR_values,estpc(NR_values,k,s))
plt.errorbar(NR_values, mean_probs, yerr=errs, fmt='o', markersize=3, capsize=4)
plt.xlabel("Number of Receptors (NR)")
plt.ylabel("Decoding Accuracy")
plt.title("Efficient Sparse Odor Decoding vs. Receptor Count")
plt.grid(True)
plt.tight_layout()
plt.savefig("binary1")
plt.show()


def phalf(s):
    return -(-4.61486830584+np.log(1/s))/s

NR = 1000
NL = 100000
k_vals = np.arange(1,50,1)
s_vals = np.linspace(0.01,0.1,49)
probs=np.zeros((49,49))
sNRs = []
ESTs = []

for i,S in enumerate(s_vals):
    sNRs.append(S*NR)
    ESTs.append(phalf(S))
    print(f"Working on {i+1}/{len(s_vals)}")
    for j,K in enumerate(k_vals):
        print(f"\tWorking on {j+1}/{len(k_vals)}")
        correct = 0
        for _ in range(10):
            odor_guess, _, odor_vec, _ = decode(NL, NR, K, S)
            correct += np.array_equal(odor_guess, odor_vec)
        probs[j][i]=(correct/100)
plt.plot(sNRs, ESTs,color='w')
plt.contourf(sNRs, k_vals, probs, levels=50, cmap='viridis')
plt.colorbar(label='P(c=c)')
plt.xlabel('s*NR')
plt.ylabel('K')
plt.savefig("binary2")
plt.show()

