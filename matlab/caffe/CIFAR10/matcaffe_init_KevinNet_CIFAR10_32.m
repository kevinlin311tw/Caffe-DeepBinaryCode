function  matcaffe_init_KevinNet_CIFAR10_32(use_gpu, model_def_file, model_file)
% matcaffe_init(model_def_file, model_file, use_gpu)
% Initilize matcaffe wrapper

if nargin < 1
  % By default use CPU
  use_gpu = 0;
end
if nargin < 2 || isempty(model_def_file)
  % By default use imagenet_deploy
  model_def_file = '/home/iis/deep/rcnn_packages/Caffe-DeepBinaryCodes/examples/mycifar10/KevinNet_v2/CIFAR10-32/KevinNet_CIFAR10_32_deploy.prototxt';
end
if nargin < 3 || isempty(model_file)
  model_file = '/home/iis/deep/rcnn_packages/Caffe-DeepBinaryCodes-models/KevinNet_CIFAR10_32_iter_50000.caffemodel';
end


%if caffe('is_initialized') == 0
  if exist(model_file, 'file') == 0
    % NOTE: you'll have to get the pre-trained ILSVRC network
    error('You need a network model file');
  end
  if ~exist(model_def_file,'file')
    % NOTE: you'll have to get network definition
    error('You need the network prototxt definition');
  end
  caffe('init', model_def_file, model_file)
%end
fprintf('Done with init\n');

% set to use GPU or CPU
if use_gpu
  fprintf('Using GPU Mode\n');
  caffe('set_mode_gpu');
else
  fprintf('Using CPU Mode\n');
  caffe('set_mode_cpu');
end
fprintf('Done with set_mode\n');

% put into test mode
caffe('set_phase_test');
