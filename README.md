# Caffe-DeepBinaryCode

Implementation of the supervised semantics-preserving deep hashing (SSDH)

Created by Kevin Lin, Huei-Fang Yang, and Chu-Song Chen at Academia Sinica, Taipei, Taiwan.

### Introduction

SSDH constructs hash functions as a latent layer in a deep network in which binary codes are learned by the optimization of an objective function defined over classification error and other desirable properties of hash codes.

This modified caffe distribution provides two objective functions, K1 and K2, to learn efficient binary hash codes. The details can be found in the following [arxiv paper](http://arxiv.org/abs/1507.00101)

