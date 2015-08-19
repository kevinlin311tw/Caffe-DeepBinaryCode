# Caffe-DeepBinaryCode

Implementation of the Supervised Semantics-preserving Deep Hashing (SSDH)

Created by Kevin Lin, Huei-Fang Yang, and Chu-Song Chen at Academia Sinica, Taipei, Taiwan.

### Introduction

We present a simple yet effective supervised deep hash approach that constructs binary hash codes from labeled data for large-scale image search. We assume that the semantic labels are governed by several latent attributes with each attribute on or off, and classification relies on these attributes. Based on this assumption, our approach, dubbed supervised semantics-preserving deep hashing (SSDH), constructs hash functions as a latent layer in a deep network and the binary codes are learned by minimizing an objective function defined over classification error and other desirable hash codes properties. With this design, SSDH has a nice characteristic that classification and retrieval are unified in a single learning model. Moreover, SSDH performs joint learning of image representations, hash codes, and classification in a point-wised manner, and thus is scalable to large-scale datasets. SSDH is simple and can be realized by a slight enhancement of an existing deep architecture for classification; yet it is effective and outperforms other hashing approaches on several benchmarks and large datasets. Compared to state-of-the-art results, SSDH achieves 26.30% (89.68% vs. 63.38%), 17.11% (89.00% vs. 71.89%) and 19.56% (31.28% vs. 11.72%) higher precisions averaged over a different number of top returned images for the CIFAR-10, NUS-WIDE, and SUN397 datasets, respectively, while the classification performance is not sacrificed.

This modified caffe distribution provides the proposed objective functions (K1 and K2) to learn efficient binary hash codes. 

The details can be found in the following [arxiv paper](http://arxiv.org/abs/1507.00101)

