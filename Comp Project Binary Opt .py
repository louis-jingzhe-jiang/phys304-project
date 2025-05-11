#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 23:10:57 2025
This notebook defined a computational model for the olfactory system by reducing the concentration of odorants
in an odor mixture to simple presence or absence of an odor and also reducing the bonding affinity of olfactory
receptors into simple bonding or not bonding, making the algorithm entirely binary. within the notebook, there is 
an encoding process, which creates a random odor vector and sensing matrix to mimic an odor mixture and random
array of odorants that can bind to different receptors.  This encoding process then creates the vector representing
the response of the olfactory receptors. This notbook also defines a decoding process in which the receptor vector and 
sensing matrix are used to work backwards and create an estimate of the original odor mixture vector. The guess is then compared
to the original vector. Using these functions, the notebook graphs the dependence of decoding accuracy on the number of 
receptors and then also the dependence of decoding accuracy on the number of odorants per odor and average number of odorants
responding to a receptor.

To run the notebook, simply hit the SHIFT and ENTER keys together. *Note: the code takes a considerable amount of time.
    

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
        #finding the location of the odorants in the rows corresponding to inactive receptors
        zero_out = np.any(sense_mat[inactive_receptors] == 1, axis=0)
        odor_guess[0, zero_out] = 0

    return odor_guess, sense_mat, odor_vec.T, recep_vec

#defining a function for the analytic solution
def estpc(Nr,k,s):
    alpha = k/NL
    return (alpha+(1-alpha)*(1-(1-s*(1-s*alpha)**(NL-1))**Nr))**NL

#defining parameters
NL = 10000
k = 10
s = 0.05
NR_values = np.arange(100, 600, 10)

mean_probs = []
errs = []


# simulating encoding and decoding for different values of NR
probs=[]
for NR in NR_values:
    #printing NR to let the user know where the code is in the process
    print(NR)
    trial_accuracies = np.zeros(10)
    #looping through 10 simulations
    for trial in range(10):
        
        #defining a value to record number of successful trials
        correct = 0
        
        #repeating 100 trials per simulation
        for _ in range(100):
            odor_guess, _, odor_vec, _ = decode(NL, NR, k, s)
            #updating whether or not the decoding was accurate
            correct += np.array_equal(odor_guess, odor_vec)
        
        #finding the mean accuracy for the trials
        trial_accuracies[trial] = correct / 100
    mean_probs.append(np.mean(trial_accuracies))
    
    #using the 10 simulations to get an error bar
    errs.append(np.std(trial_accuracies))


# Ploting analytical solution
plt.plot(NR_values,estpc(NR_values,k,s))

#plotting binary decoder
plt.errorbar(NR_values, mean_probs, yerr=errs, fmt='o', markersize=3, capsize=4)
plt.xlabel("Number of Receptors (NR)")
plt.ylabel("Decoding Accuracy")
plt.title("Efficient Sparse Odor Decoding vs. Receptor Count")
plt.grid(True)
plt.tight_layout()
plt.savefig("binary1")
plt.show()

#defining a function to plot the P(c=c)=0.5 case 
def phalf(s):
    return -(-4.61486830584+np.log(1/s))/s

#defining parameters
NR = 1000
NL = 100000

#making values for k and s to iterate through
k_vals = np.arange(1,50,1)
s_vals = np.linspace(0.01,0.1,49)
probs=np.zeros((49,49))

#making blank lists for s*NR values and analyic solution
sNRs = []
ESTs = []

#iterating through the s and v value lists
for i,S in enumerate(s_vals):
    
    #getting s*NR values
    sNRs.append(S*NR)
    
    #calculating analytic solution for p(c=c)=0.5
    ESTs.append(phalf(S))
    
    #printing to show where the code is in the data making process
    print(f"Working on {i+1}/{len(s_vals)}")
    for j,K in enumerate(k_vals):
        
        print(f"\tWorking on {j+1}/{len(k_vals)}")
        
        #defining a value to record number of successful trials
        correct = 0
        
        #averaging the success of the decoding algorithm for 10 trials at each value of k and s
        for _ in range(10):
            odor_guess, _, odor_vec, _ = decode(NL, NR, K, S)
            correct += np.array_equal(odor_guess, odor_vec)
        probs[j][i]=(correct/10)
        
#plotting the analytic solution for p(c=c)=0.5
plt.plot(sNRs, ESTs,color='w')

#plotting the trial data for the decoder
plt.contourf(sNRs, k_vals, probs, levels=50, cmap='viridis')
plt.colorbar(label='P(c=c)')
plt.xlabel('s*NR')
plt.ylabel('K')
plt.savefig("binary2")
plt.show()

