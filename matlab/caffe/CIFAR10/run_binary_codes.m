%{ 
[scores_test , list_im_test] = matcaffe_batch_KevinNet_CIFAR10_32('cifar10_files_test.txt', 1);
 binary32_test = (scores_test>0.5);
 [scores_train , list_im_train] = matcaffe_batch_KevinNet_CIFAR10_32('cifar10_files_train.txt', 1);
 binary32_train = (scores_train>0.5);
 %}
%{
  [scores_test , list_im_test] = matcaffe_batch_KevinNet_CIFAR10_12('cifar10_files_test.txt', 1);
 binary12_test = (scores_test>0.5);
 save(['binary12_test.mat'],'binary12_test','-v7.3');
 [scores_train , list_im_train] = matcaffe_batch_KevinNet_CIFAR10_12('cifar10_files_train.txt', 1);
 binary12_train = (scores_train>0.5);
 save(['binary12_train.mat'],'binary12_train','-v7.3');
 
 
  [scores_test , list_im_test] = matcaffe_batch_KevinNet_CIFAR10_48('cifar10_files_test.txt', 1);
 binary48_test = (scores_test>0.5);
 save(['binary48_test.mat'],'binary48_test','-v7.3');
 [scores_train , list_im_train] = matcaffe_batch_KevinNet_CIFAR10_48('cifar10_files_train.txt', 1);
 binary48_train = (scores_train>0.5);
 save(['binary48_train.mat'],'binary48_train','-v7.3');
 %}
%{
  [scores_test , list_im_test] = matcaffe_batch_KevinNet_CIFAR10_64('cifar10_files_test.txt', 1);
 binary64_test = (scores_test>0.5);
 save(['binary64_test.mat'],'binary64_test','-v7.3');
 [scores_train , list_im_train] = matcaffe_batch_KevinNet_CIFAR10_64('cifar10_files_train.txt', 1);
 binary64_train = (scores_train>0.5);
 save(['binary64_train.mat'],'binary64_train','-v7.3');
 %}
 %cifar_eval( list_im_train, trn_label, binary12_train, list_im_test, tst_label, binary12_test, 12);
 %cifar_eval( list_im_train, trn_label, binary32_train, list_im_test, tst_label, binary32_test, 32);
 %cifar_eval( list_im_train, trn_label, binary48_train, list_im_test, tst_label, binary48_test, 48);
 %cifar_eval( list_im_train, trn_label, binary64_train, list_im_test, tst_label, binary64_test, 64);
 
 
 [scores_test , list_im_test] = matcaffe_batch_AlexNet_CIFAR10_4096('cifar10_files_test.txt', 1);
 alex4096_test = (scores_test>0.5);
 save(['alex4096_test.mat'],'alex4096_test','-v7.3');
 [scores_train , list_im_train] = matcaffe_batch_AlexNet_CIFAR10_4096('cifar10_files_train.txt', 1);
 alex4096_train = (scores_train>0.5);
 save(['alex4096_train.mat'],'alex4096_train','-v7.3');
 cifar_eval( list_im_train, trn_label, alex4096_train, list_im_test, tst_label, alex4096_test, 4096);