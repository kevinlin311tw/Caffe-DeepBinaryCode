_FYI: **We will provide the training scripts and user guide soon!!!** If you'd like to run the training code now, please make sure you already have **the dataset (leveldb or lmdb, w/h=256/256)** and **ImageNet pre-trained weights**_

---

# Caffe-DeepBinaryCode

Implementation of the Supervised Semantics-preserving Deep Hashing (SSDH)

Created by Huei-Fang Yang, Kevin Lin, and Chu-Song Chen at Academia Sinica, Taipei, Taiwan.

---

### Introduction

We present a simple yet effective supervised deep hash approach that constructs binary hash codes from labeled data for large-scale image search. SSDH constructs hash functions as a latent layer in a deep network and the binary codes are learned by minimizing an objective function defined over classification error and other desirable hash codes properties. Compared to state-of-the-art results, SSDH achieves 26.30% (89.68% vs. 63.38%), 17.11% (89.00% vs. 71.89%) and 19.56% (31.28% vs. 11.72%) higher precisions averaged over a different number of top returned images for the CIFAR-10, NUS-WIDE, and SUN397 datasets, respectively.

This modified caffe distribution provides our objective function to learn efficient binary hash codes. 

The details can be found in the following [arXiv preprint.](http://arxiv.org/abs/1507.00101)

---

### Citing the deep hashing work

If you find our work useful in your research, please consider citing:

    Supervised Learning of Semantics-Preserving Hashing via Deep Neural Networks for Large-Scale Image Search
    Huei-Fang Yang, Kevin Lin, Chu-Song Chen
    arXiv preprint arXiv:1507.00101



