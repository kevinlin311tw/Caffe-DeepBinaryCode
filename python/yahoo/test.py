#!/usr/bin/env python

import sys
import PIL
from PIL import Image

caffe_root = '../../' 

sys.path.insert(0, caffe_root + 'python')
import caffe
import extractBatch
import numpy as np



def doFeaExtraction(filename):


    caffeModelDefinitionFileName = 'models/KevinNet_Yahoo1M_128_deploy.prototxt';
    caffePretrainedModelFileName = 'models/KevinNet_Yahoo1M_128_iter_750000.caffemodel';

    imageDims = [256,256];
    gpuId = -1;
    meanFileName = 'ilsvrc_2012_mean.npy';
    inputScale = None;
    rawScale = 255.0;
    channelSwap = [2,1,0];

    featureExtractor = extractBatch.init(caffeModelDefinitionFileName,
                                                  caffePretrainedModelFileName,
                                                  image_dims=imageDims,
                                                  gpu_id=gpuId,
                                                  mean_file=np.load(meanFileName),
                                                  input_scale=inputScale,
                                                  raw_scale=rawScale,
                                                  channel_swap=channelSwap)
   
    queryFeatureVectors = extractBatch.extractFile(filename, featureExtractor, True, layer_name = 'fc7')

    print queryFeatureVectors;
    print 'len(queryFeatureVectors):' + str(len(queryFeatureVectors[0]));


def main(argv):

    filename = sys.argv[1]

    doFeaExtraction(filename);



if __name__ == '__main__':
    main(sys.argv)
