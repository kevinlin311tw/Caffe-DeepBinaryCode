_FYI: This project is a **work in progress**. If you'd like to run the training code, please make sure you already have the dataset (leveldb/lmdb) and ImageNet pre-train model!!!!!!_

---


# Caffe-DeepBinaryCode

Implementation of the Supervised Semantics-preserving Deep Hashing (SSDH)

Created by Kevin Lin, Huei-Fang Yang, and Chu-Song Chen at Academia Sinica, Taipei, Taiwan.

### Introduction

We present a simple yet effective supervised deep hash approach that constructs binary hash codes from labeled data for large-scale image search. SSDH constructs hash functions as a latent layer in a deep network and the binary codes are learned by minimizing an objective function defined over classification error and other desirable hash codes properties. Compared to state-of-the-art results, SSDH achieves 26.30% (89.68% vs. 63.38%), 17.11% (89.00% vs. 71.89%) and 19.56% (31.28% vs. 11.72%) higher precisions averaged over a different number of top returned images for the CIFAR-10, NUS-WIDE, and SUN397 datasets, respectively.

This modified caffe distribution provides the proposed objective function to learn efficient binary hash codes. 

The details can be found in the following [arXiv preprint.](http://arxiv.org/abs/1507.00101)


### Citing the deep hashing work

If you find our works useful in your research, please consider citing:

    Supervised Learning of Semantics-Preserving Hashing via Deep Neural Networks for Large-Scale Image Search
    Huei-Fang Yang, Kevin Lin, Chu-Song Chen
    arXiv preprint arXiv:1507.00101
