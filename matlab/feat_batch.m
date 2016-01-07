function [scores, list_im]= feat_batch (use_gpu, net_model, net_weights, list_im, dim)
caffe.reset_all();
if nargin < 1
  % By default use CPU
  use_gpu = 0;
end
if nargin < 2 || isempty(net_model)
  % By default use imagenet_deploy
  net_model = '/home/titan/hfyang/hashing/training/NUS-WIDE/NUS-KevinNet-48/KevinNet_NUS_48_deploy.prototxt';
end
if nargin < 3 || isempty(net_weights)
  % By default use caffe reference model
  net_weights = '/home/titan/hfyang/hashing/training/NUS-WIDE/NUS-KevinNet-48/models-20150720/KevinNet_NUS_48_iter_50000';
end
if ischar(list_im)
    %Assume it is a file contaning the list of images
    filename = list_im;
    list_im = read_cell(filename);
end
% Set caffe mode
if exist('use_gpu', 'var') && use_gpu
  caffe.set_mode_gpu();
  gpu_id = 0;  % we will use the first gpu in this demo
  caffe.set_device(gpu_id);
else
  caffe.set_mode_cpu();
end

% Initialize the network using BVLC CaffeNet for image classification
% Weights (parameter) file needs to be downloaded from Model Zoo.
phase = 'test'; % run with phase test (so that dropout isn't applied)

if ~exist(net_weights, 'file')
    error('%s does not exist.', net_weights);
end
if ~exist(net_model, 'file')
    error('%s does not exist.', net_model);
end

% Initialize a network
net = caffe.Net(net_model, net_weights, phase);

% load mean file
d = load('/home/iis/adsc/caffe-new-cbd-udnn/matlab/+caffe/imagenet/ilsvrc_2012_mean.mat');
mean_data = d.mean_data;

batch_size = 10;

% prepare input
num_images = length(list_im);
scores = zeros(dim,num_images,'single');
num_batches = ceil(length(list_im)/batch_size);
initic=tic;
for bb = 1 : num_batches
    batchtic = tic;
    range = 1+batch_size*(bb-1):min(num_images,batch_size * bb);
    tic
    input_data = {prepare_batch(list_im(range),mean_data,batch_size)};
    toc, tic
    fprintf('Batch %d out of %d %.2f%% Complete ETA %.2f seconds\n',...
        bb,num_batches,bb/num_batches*100,toc(initic)/bb*(num_batches-bb));
    %output_data = caffe('forward', {input_data});  
    output_data = net.forward(input_data);
    toc
    
    output_data = squeeze(output_data{1});
    scores(:,range) = output_data(:,mod(range-1,batch_size)+1);
    toc(batchtic)
end
toc(initic);
%if exist('filename', 'var')
%    save([filename '.probs.mat'],'list_im','scores','-v7.3');
%end

% call caffe.reset_all() to reset caffe
caffe.reset_all();
end
