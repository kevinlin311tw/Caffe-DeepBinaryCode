# Caffe-DeepBinaryCode

Supervised Learning of Semantics-Preserving Deep Hashing (SSDH)

Created by Kevin Lin, Huei-Fang Yang, and Chu-Song Chen at Academia Sinica, Taipei, Taiwan.


## Introduction

<img src="https://www.csie.ntu.edu.tw/~r01944012/ssdh_intro.png" width="800">

We present a simple yet effective supervised deep hash approach that constructs binary hash codes from labeled data for large-scale image search. SSDH constructs hash functions as a latent layer in a deep network and the binary codes are learned by minimizing an objective function defined over classification error and other desirable hash codes properties. Compared to state-of-the-art results, SSDH achieves 26.30% (89.68% vs. 63.38%), 17.11% (89.00% vs. 71.89%) and 19.56% (31.28% vs. 11.72%) higher precisions averaged over a different number of top returned images for the CIFAR-10, NUS-WIDE, and SUN397 datasets, respectively.

The details can be found in the following [arXiv preprint.](http://arxiv.org/abs/1507.00101)
Presentation slide [PDF](http://www.csie.ntu.edu.tw/~r01944012/deepworkshop-slide.pdf)


### Citing the deep hashing work

If you find our work useful in your research, please consider citing:

    Supervised Learning of Semantics-Preserving Hashing via Deep Neural Networks for Large-Scale Image Search
    Huei-Fang Yang, Kevin Lin, Chu-Song Chen
    arXiv preprint arXiv:1507.00101

## Prerequisites

  0. MATLAB (tested with 2012b on 64-bit Linux)
  0. Caffe's [prerequisites](http://caffe.berkeleyvision.org/installation.html#prequequisites)


## Install Caffe-cvprw15

Adjust Makefile.config and simply run the following commands:

    $ make all -j8
    $ make test -j8
    $ make matcaffe
    $ ./prepare.sh

For a faster build, compile in parallel by doing `make all -j8` where 8 is the number of parallel threads for compilation (a good choice for the number of threads is the number of cores in your machine).

## Demo
 
Launch matlab and run `demo.m`
    
    >> demo

This demo will generate 48-bits binary codes for each image using the proposed SSDH.

<img src="https://www.csie.ntu.edu.tw/~r01944012/ssdh_demo.png" width="200">


## Retrieval evaluation on CIFAR10

Launch matalb and run `run_cifar10.m` to perform the evaluation of `precision at k` and `mean average precision at k`. We set `k=1000` in the experiments. The bit length of binary codes is `48`. This process takes around 12 minutes.
    
    >> run_cifar10


Then, you will get the `mAP` result as follows. 

    >> MAP = 0.899731

Moreover, simply run the following commands to generate the `precision at k` curves:

    $ cd analysis
    $ gnuplot plot-p-at-k.gnuplot 

You will reproduce the precision curves with respect to different number of top retrieved samples when the 48-bit hash codes are
used in the evaluation.
 
## Train your own model on CIFAR10

Simply run the following command to train SSDH:


    $ ./examples/SSDH/train.sh


The training process takes roughly 2~3 hours on a desktop with Titian X GPU.
You will finally get your model named `SSDH48_iter_xxxxxx.caffemodel` under folder `/examples/SSDH/`

To use the model, modify the `model_file` in `demo.m` to link to your model:

```
    model_file = './YOUR/MODEL/PATH/filename.caffemodel';
```

Launch matlab, run `demo.m` and enjoy!
    
    >> demo

## Contact

Please feel free to leave suggestions or comments to Kevin Lin (kevinlin311.tw@iis.sinica.edu.tw), Huei-Fang Yang (hfyang@citi.sinica.edu.tw) or Chu-Song Chen (song@iis.sinica.edu.tw)



