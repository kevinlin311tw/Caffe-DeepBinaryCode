% Demo of binary codes and deep feature extraction  
% Modify 'test_file_list' and get the features of your images!

close all;
clear;

% -----------------------------------------------------------
% 48-bits binary codes extraction
%
% input
%   	img_list.txt:  list of images files 
% output
%   	binary_codes: 48 x num_images output binary vector
%   	list_im: the corresponding image path
%
% ----- settings start here -----
% set 1 to use gpu, and 0 to use cpu
use_gpu = 1;
% binary code length
feat_len = 48;
% models
model_file = './models/SSDH/SSDH48_iter_50000.caffemodel';
% model definition
model_def_file = './models/SSDH/deploy.prototxt';
% caffe mode setting
phase = 'test'; % run with phase test (so that dropout isn't applied)
% input data
test_file_list = 'img_list.txt';
% ------ settings end here ------

% Extract binary hash codes
[feat_test, list_im]= feat_batch(use_gpu, model_def_file, model_file, test_file_list, feat_len);
binary_codes = (feat_test>0.5);
save('binary48.mat','binary_codes','list_im','-v7.3');


% Visualization
figure(1),
set(gcf, 'Position'); 
for i=1:4
    image = sprintf('.%s',list_im{i});
    codes = num2str(binary_codes(:,i)');
    codes = sprintf('binary codes: %s',codes);
    subplot(4,1,i), imshow(image); title(codes);
end

