# Caffe-DeepBinaryCode

Supervised Learning of Semantics-Preserving Deep Hashing (SSDH)

Created by Kevin Lin, Huei-Fang Yang, and Chu-Song Chen at Academia Sinica, Taipei, Taiwan.


## Introduction

This paper presents a simple yet effective supervised deep hash approach that constructs binary hash codes from labeled data for large-scale image search. We assume that the semantic labels are governed by several latent attributes with each attribute on or off, and classification relies on these attributes. Based on this assumption, our approach, dubbed supervised semantics-preserving deep hashing (SSDH), constructs hash functions as a latent layer in a deep network and the binary codes are learned by minimizing an objective function defined over classification error and other desirable hash codes properties. With this design, SSDH has a nice characteristic that classification and retrieval are unified in a single learning model. Moreover, SSDH performs joint learning of image representations, hash codes, and classification in a point-wised manner, and thus is scalable to large-scale datasets. SSDH is simple and can be realized by a slight enhancement of an existing deep architecture for classification; yet it is effective and outperforms other hashing approaches on several benchmarks and large datasets. Compared with state-of-the-art approaches, SSDH achieves higher retrieval accuracy, while the classification performance is not sacrificed.

<img src="https://www.csie.ntu.edu.tw/~r01944012/ssdh_intro.png" width="800">

The details can be found in the following [arXiv preprint.](http://arxiv.org/abs/1507.00101)
Presentation slide can be found [here](http://www.csie.ntu.edu.tw/~r01944012/deepworkshop-slide.pdf)


## Citing the deep hashing work

If you find our work useful in your research, please consider citing:

    Supervised Learning of Semantics-Preserving Hash via Deep Convolutional Neural Networks
    Huei-Fang Yang, Kevin Lin, Chu-Song Chen
    IEEE Transactions on Pattern Analysis and Machine Intelligence (​TPAMI), 2017

    Supervised Learning of Semantics-Preserving Hashing via Deep Neural Networks for Large-Scale Image Search
    Huei-Fang Yang, Kevin Lin, Chu-Song Chen
    arXiv preprint arXiv:1507.00101



## Prerequisites

  0. MATLAB (tested with 2012b on 64-bit Linux)
  0. Caffe's [prerequisites](http://caffe.berkeleyvision.org/installation.html#prequequisites)


## Install Caffe-DeepBinaryCode

Adjust Makefile.config and simply run the following commands:

    $ make all -j8
    $ make test -j8
    $ make matcaffe
    $ ./prepare.sh

For a faster build, compile in parallel by doing `make all -j8` where 8 is the number of parallel threads for compilation (a good choice for the number of threads is the number of cores in your machine).

## Demo
 
Launch matlab and run `demo.m`. This demo will generate 48-bits binary codes for each image using the proposed SSDH.
    
    >> demo


<img src="https://www.csie.ntu.edu.tw/~r01944012/ssdh_demo.png" width="350">


## Retrieval evaluation on CIFAR10

Launch matalb and run `run_cifar10.m` to perform the evaluation of `precision at k` and `mean average precision (mAP) at k`. In this CIFAR10 experiment, we employ all the test images (`10,000` images) as the query set, and we select all the training images (`50,000` images) to form the database. We computed mAP based on the entire retrieval list, thus we set `k = 50,000` in this experiment. The bit length of binary codes is `48`. This process takes around 12 minutes.
    
    >> run_cifar10


Then, you will get the `mAP` result as follows. 

    >> MAP = 0.913361

Moreover, simply run the following commands to generate the `precision at k` curves:

    $ cd analysis
    $ gnuplot plot-p-at-k.gnuplot 

You will reproduce the precision curves with respect to different number of top retrieved samples when the 48-bit hash codes are
used in the evaluation.
 
## Train SSDH on CIFAR10

Simply run the following command to train SSDH:

    $ cd /examples/SSDH
    $ ./train.sh


After 50,000 iterations, the top-1 error rate is around 10% on the test set of CIFAR10 dataset:
```
I1221 16:27:44.764175  2985 solver.cpp:326] Iteration 50000, loss = -0.10567
I1221 16:27:44.764205  2985 solver.cpp:346] Iteration 50000, Testing net (#0)
I1221 16:27:58.907842  2985 solver.cpp:414]     Test net output #0: accuracy = 0.8989
I1221 16:27:58.907877  2985 solver.cpp:414]     Test net output #1: loss: 50%-fire-rate = 0.000621793 (* 1 = 0.000621793 loss)
I1221 16:27:58.907886  2985 solver.cpp:414]     Test net output #2: loss: classfication-error = 0.369317 (* 1 = 0.369317 loss)
I1221 16:27:58.907892  2985 solver.cpp:414]     Test net output #3: loss: forcing-binary = -0.114405 (* 1 = -0.114405 loss)
I1221 16:27:58.907897  2985 solver.cpp:331] Optimization Done.
I1221 16:27:58.907902  2985 caffe.cpp:214] Optimization Done.
```



The training process takes roughly 2~3 hours on a desktop with Titian X GPU. You will finally get your model named `SSDH48_iter_xxxxxx.caffemodel` under folder `/examples/SSDH/`

To use the model, modify the `model_file` in `demo.m` to link to your model:

```
    model_file = './YOUR/MODEL/PATH/filename.caffemodel';
```

Launch matlab, run `demo.m` and enjoy!
    
    >> demo

## Train SSDH on another dataset

It should be easy to train the model using another dataset as long as that dataset has label annotations.
 
  0. Convert your training/test set into leveldb/lmdb format using `create_imagenet.sh`.
  0. Modify the `source` in `/example/SSDH/train_val.prototxt` to link to your training/test set.
  0. Run `./examples/SSDH/train.sh`, and start training on your dataset.


## Resources

**Note**: This documentation may contain links to third party websites, which are provided for your convenience only. Third party websites may be subject to the third party’s terms, conditions, and privacy statements.

If `./prepare.sh` fails to download data, you may manually download the resouces from:

0. 48-bit SSDH model: [MEGA](https://mega.nz/#!9JMBlCaS!zsTl7eZRMdi25gkLWpj_Uv8LfN_2gQ-UF8OBMhio_3s), [DropBox](https://www.dropbox.com/s/6iqyz1mdhadhzbu/SSDH48_iter_50000.caffemodel?dl=0), [BaiduYun coming soon]

0. CIFAR10 dataset (jpg format): [MEGA](https://mega.nz/#!RENV1bhZ!x0uFnAkqUSTJzKr6HzeeNV9mtDjlgQ0x6ZaXfpxbJkw), [DropBox](https://www.dropbox.com/s/f7q3bbgvat2q1u2/cifar10-dataset.zip?dl=0), [BaiduYun](http://pan.baidu.com/s/1pKsSK7h)

0. AlexNet pretrained networks: [MEGA](https://mega.nz/#!UZ0VGIYB!y2crhbo89S9hYLv5TyHLXXB5Sus8ZkpUzTNkeUPkfU4), [DropBox](https://www.dropbox.com/s/nlggnj47xxdmwkb/bvlc_reference_caffenet.caffemodel?dl=0), [BaiduYun](http://pan.baidu.com/s/1qWRMy4G)

## FAQ

Q: I have followed the instructions in README, and ran the evaluation code. As shown in README that I will get the mAP around `90%`, however, I can only get about `10%` mAP. Could you please give me some suggestions?

A: You may have this problem if you didn’t launch matlab at caffe's root folder, which will automatically include important folders into PATH. 
Two ways to solve this problem: First, run `startup.m` before you run `run_cifar10.m`. Second, open `./matlab/feat_batch.m` and change line 36 `d = load('./matlab/+caffe/imagenet/ilsvrc_2012_mean.mat');` to `d = load('THE-PATH-OF-THIS-REPO-IN-YOU-COMPUTER/matlab/+caffe/imagenet/ilsvrc_2012_mean.mat');`. Then run `run_cifar10.m`.


## Contact

Please feel free to leave suggestions or comments to Kevin Lin (kevinlin311.tw@iis.sinica.edu.tw), Huei-Fang Yang (hfyang@citi.sinica.edu.tw) or Chu-Song Chen (song@iis.sinica.edu.tw)



