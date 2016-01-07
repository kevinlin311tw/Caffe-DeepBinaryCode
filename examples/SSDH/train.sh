./build/tools/caffe train -solver  ./examples/SSDH/solver.prototxt -weights ./models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel -gpu 0 2>&1 | tee ./examples/SSDH/log.txt
