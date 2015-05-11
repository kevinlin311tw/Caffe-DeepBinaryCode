 [scores_test , list_im_test] = matcaffe_batch_KevinNet_CIFAR10_32('cifar10_files_test.txt', 1);
 binary32_test = (scores_test>0.5);
 [scores_train , list_im_train] = matcaffe_batch_KevinNet_CIFAR10_32('cifar10_files_train.txt', 1);
 binary32_train = (scores_train>0.5);