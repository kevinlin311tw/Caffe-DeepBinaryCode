#!/usr/bin/env python
"""
extractBatch.py extracts caffe features, callable from the command line.

By default it configures and runs the Caffe reference ImageNet model.
"""
import numpy as np
import os
import sys
import argparse
import glob
import time

import caffe


def init(model_def, pretrained_model, image_dims, gpu_id, mean_file,
            input_scale, raw_scale, channel_swap):
    feature_extractor = caffe.Extractor(model_def, pretrained_model,
            image_dims=image_dims, gpu_id=gpu_id, mean=mean_file,
            input_scale=input_scale, raw_scale=raw_scale,
            channel_swap=channel_swap)
    return feature_extractor

def extract(inputs, feature_extractor, center_only, layer_name = 'fc7'):
    features = feature_extractor.extract(inputs, center_only, layer_name = layer_name)
    return features    

def getFeatureStr(label, feature_vector, sparse=False):
    feature_str = label

    for fid, value in enumerate(feature_vector):
        if sparse:
            if value < 0.0000000001:
                continue

        feature_str += ' ' + str(fid + 1) + ':' + str(value)

    return feature_str

def extractFile(fileName, feature_extractor, center_only, layer_name = 'fc7'):
    features = extract([caffe.io.load_image(fileName)], feature_extractor, center_only, layer_name)
    return features


def main(argv):
    pycaffe_dir = os.path.dirname(__file__)

    parser = argparse.ArgumentParser()
    # Required arguments: input and output files.
    parser.add_argument(
        "input_file",
        help="Input image, directory, or npy."
    )
    parser.add_argument(
        "output_file",
        help="Output npy filename."
    )
    # Optional arguments.
    parser.add_argument(
        "--model_def",
        default=os.path.join(pycaffe_dir,
                "../models/bvlc_reference_caffenet/deploy.prototxt"),
        help="Model definition file."
    )
    parser.add_argument(
        "--pretrained_model",
        default=os.path.join(pycaffe_dir,
                "../models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel"),
        help="Trained model weights file."
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        help="Switch for gpu computation."
    )
    parser.add_argument(
        "--center_only",
        action='store_true',
        help="Switch for prediction from center crop alone instead of " +
             "averaging predictions across crops (default)."
    )
    parser.add_argument(
        "--images_dim",
        default='256,256',
        help="Canonical 'height,width' dimensions of input images."
    )
    parser.add_argument(
        "--mean_file",
        default=os.path.join(pycaffe_dir,
                             'caffe/imagenet/ilsvrc_2012_mean.npy'),
        help="Data set image mean of H x W x K dimensions (numpy array). " +
             "Set to '' for no mean subtraction."
    )
    parser.add_argument(
        "--input_scale",
        type=float,
        help="Multiply input features by this scale to finish preprocessing."
    )
    parser.add_argument(
        "--raw_scale",
        type=float,
        default=255.0,
        help="Multiply raw input by this scale before preprocessing."
    )
    parser.add_argument(
        "--channel_swap",
        default='2,1,0',
        help="Order to permute input channels. The default converts " +
             "RGB -> BGR since BGR is the Caffe default by way of OpenCV."
    )
    parser.add_argument(
        "--ext",
        default='jpg',
        help="Image file extension to take as input when a directory " +
             "is given as the input file."
    )
    parser.add_argument(
        "--prefix",
        default='./',
        help="Prefix for the image file paths specified in the input file."
    )
    parser.add_argument(
        "--batch_size",
        default='256',
        help="Size of mini batch to pas to caffe."
    )
    parser.add_argument(
        "--layer_name",
        default='fc7',
        help="Layer name for the features you want to extract."
    )
    parser.add_argument(
        "--sparse",
        action='store_true',
        help="Output sparse feature vector."
    )
    args = parser.parse_args()

    image_dims = [int(s) for s in args.images_dim.split(',')]

    mean, channel_swap = None, None
    if args.mean_file:
        mean = np.load(args.mean_file)
    if args.channel_swap:
        channel_swap = [int(s) for s in args.channel_swap.split(',')]

    # Make feature extractor.
    #feature_extractor = caffe.Extractor(args.model_def, args.pretrained_model,
    #        image_dims=image_dims, gpu_id=args.gpu_id, mean=mean,
    #        input_scale=args.input_scale, raw_scale=args.raw_scale,
    #        channel_swap=channel_swap)

    feature_extractor = init(args.model_def, args.pretrained_model,
            image_dims=image_dims, gpu_id=args.gpu_id, mean_file=mean,
            input_scale=args.input_scale, raw_scale=args.raw_scale,
            channel_swap=channel_swap)

    if args.gpu_id != -1:
        print 'GPU mode'
    else:
        print 'CPU mode'

    if args.center_only:
        print 'Center only'
    else:
        print 'Oversample'

    if args.sparse:
        print 'Sparse vectors'
    else:
        print 'Dense vectors'

    print ' '

    # Load numpy array (.npy), directory glob (*.jpg), or image file.
    args.input_file = os.path.expanduser(args.input_file)

    if args.input_file.endswith('npy'):
        inputs = np.load(args.input_file)
    elif os.path.isdir(args.input_file):
        inputs =[caffe.io.load_image(im_f)
                 for im_f in glob.glob(args.input_file + '/*.' + args.ext)]
    elif args.input_file.endswith('txt'):
        # read image file names and classIds into an array first
        input_files = [line.strip().split(' ') for line in open(args.input_file, 'r')]
    else:
        inputs = [caffe.io.load_image(args.input_file)]

    
    if not args.input_file.endswith('txt'):
        print "Classifying %d inputs." % len(inputs)

        # extract features.
        start = time.time()

        #features = feature_extractor.extract(inputs, not args.center_only, layer_name=args.layer_name)
        features = extract(inputs, feature_extractor, not args.center_only, layer_name=args.layer_name)

        print "Done in %.2f s." % (time.time() - start)

        # Save
        np.save(args.output_file, features)

    else:

        batch_size = int(args.batch_size)
        output_stream = open(args.output_file, 'w')

        print "Extracting features for %d inputs in batches of %d." % (len(input_files), batch_size)

        # Extract features in batches.
        start = time.time()

        while(input_files):
            inputs = [caffe.io.load_image(args.prefix + image_file[0]) for image_file in input_files[:batch_size]]

            #features = feature_extractor.extract(inputs, not args.center_only, args.layer_name)
            features = extract(inputs, feature_extractor, not args.center_only, args.layer_name)

            # Save to file
            for index, feature_vector in enumerate(features):
                feature_str = getFeatureStr(input_files[index][1], feature_vector, args.sparse)
                output_stream.write(feature_str + '\n')
                
            input_files = input_files[batch_size:]


        print "Done in %.2f s." % (time.time() - start)

        output_stream.close()


if __name__ == '__main__':
    main(sys.argv)
