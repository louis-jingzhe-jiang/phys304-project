#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 21:40:52 2025

@author: willmurdock
"""
import numpy as np
import random
import matplotlib.pyplot as plt


def make_weighted_sense_mat(NL,NR,k,s):
    #making the vector for the odor using k as number of odorants per odor
    odor_vec = np.zeros((NL,1))
    for odorant in random.sample(range(NL), k):
        odor_vec[odorant] = random.random()
   
    #making the sensitivity matrix with NR receptors and NL possible odorants per odor
    mask = np.random.rand(NR, NL) < s
    sense_mat = np.zeros((NR, NL))
    sense_mat[mask] = np.exp(np.random.uniform(np.log(0.01), np.log(1), size=np.count_nonzero(mask)))
    
    #creating the vector of receptors that bind vs. not bind
    recep_vec = sense_mat @ odor_vec
        
    return sense_mat,odor_vec,recep_vec


def weighted_decode(NL,NR,k,s):
    #defining the sense matrix, odor vector, and receptor vector using the function from before
    sense_mat,odor_vec,recep_vec = make_weighted_sense_mat(NL,NR,k,s)
    
    #defining a vector of zeros as a start for our decoding guess for the odor
    # Start with all 1s and zero out odorants that are detected by inactive receptors
    odor_guess = np.ones((1, NL), dtype=int)
    inactive_receptors = np.where(recep_vec == 0)[0]
    if len(inactive_receptors) > 0:
        
        zero_out = np.any(sense_mat[inactive_receptors] > 0, axis=0)
        active_cols = ~zero_out
        active_rows = np.ones(sense_mat.shape[0], dtype=bool)
        active_rows[inactive_receptors] = False
        
        sense_mat_new = sense_mat[active_rows][:, active_cols]
        
        recep_vec_new = recep_vec[active_rows]
        
        odor_guess = np.linalg.pinv(sense_mat_new) @ recep_vec_new
        odor_vec=odor_vec[odor_vec != 0]
        
    return odor_guess.reshape(-1,1),sense_mat,odor_vec.reshape(-1,1),recep_vec

odor_guess,sense_mat,odor_vec,recep_vec = weighted_decode(10000,500,10,0.05)



# Parameters
NL = 10000
k = 10
s = 0.05
NR_values = np.arange(100, 600, 10)

mean_probs = []
errs = []

probs=[]
for NR in NR_values:
    trial_accuracies = np.zeros(10)
    for trial in range(10):
        correct = 0
        for _ in range(100):
            odor_guess, sense_mat, odor_vec, recep_vec = weighted_decode(NL, NR, k, s)
            if odor_guess.shape == odor_vec.shape:
                correct += np.allclose(odor_guess, odor_vec, atol=1e-7)
        trial_accuracies[trial] = correct / 100
    mean_probs.append(np.mean(trial_accuracies))
    errs.append(np.std(trial_accuracies))

def exppc(Nr):
    return

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
k_vals = np.arange(1,50,1)
s_vals = np.linspace(0.01,0.1,49)
probs=np.zeros((49,49))
sNRs = []

for i,S in enumerate(s_vals):
    print(k_vals[i])
    sNRs.append(S*NR)
    for j,K in enumerate(k_vals):
        correct = 0
        for _ in range(10):
            odor_guess, _, odor_vec, _ = weighted_decode(NL, NR, K, S)
            if odor_guess.shape == odor_vec.shape:
                correct += np.allclose(odor_guess, odor_vec, atol=1e-7)
        probs[j][i]=(correct/100)
plt.contourf(sNRs, k_vals, probs, levels=50, cmap='viridis')
plt.colorbar(label='P(c=c)')
plt.xlabel('s*NR')
plt.ylabel('K')
plt.show()
'''