function [net, info] = finetune_cell(varargin)
opts.rootPath = '../data/';
opts.trainset = [1,2,3];
opts.testset = [0];
opts.whichresnet = 50;
opts.ininallabel = [];
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.expDir = fullfile(opts.rootPath, 'models/', ...
    sprintf('cell%dres-%s', opts.whichresnet, num2str(opts.trainset))) ;
opts.modelpath =  fullfile(opts.rootPath, 'models/', sprintf('imagenet-resnet-%d-dag.mat', opts.whichresnet));
% opts.modelpath =  sprintf('imagenet-resnet-%d-dag.mat', opts.whichresnet);

opts.dataDir = fullfile(opts.rootPath, '\cells');
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.whitenData = true ;
opts.contrastNormalization = true ;
opts.train = struct('gpus', 1) ;
opts = vl_argparse(opts, varargin) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end

if exist(opts.imdbPath, 'file')
    imdb = load(opts.imdbPath) ;
else
    imdb = cell_get_database(opts);
    mkdir(opts.expDir) ;
    save(opts.imdbPath, '-struct', 'imdb');
end

if ~isempty(opts.ininallabel)
    imdb.images.label(ismember(imdb.images.set, opts.testset))=opts.ininallabel;
end

net = cnn_init_res('modelPath', opts.modelpath, 'numcls', numel(imdb.meta.classes)) ;
net.meta.classes.name = imdb.meta.classes(:)' ;
trainfn = @cnn_train_dag ;

[net, info] = trainfn(net, imdb, getBatch(opts), ...
    'expDir', opts.expDir, ...
    net.meta.trainOpts, ...
    opts.train, ...
    'train', find(ismember(imdb.images.set, opts.trainset)),...
    'val', find(ismember(imdb.images.set, opts.testset))) ;
end

% -------------------------------------------------------------------------
function fn = getBatch(opts)
    bopts = struct('numGpus', numel(opts.train.gpus), 'modelType', 'res') ;
    fn = @(x,y) getDagNNBatch(bopts,x,y) ;
end

function net = cnn_init_res(varargin)
opts.numcls = 1000;
opts.modelPath = 'imagenet-resnet-101-dag.mat';
opts = vl_argparse(opts, varargin) ;
net = load(opts.modelPath);
net.meta.inputSize = [224 224 3];
net.meta.trainOpts.learningRate = [0.001 * ones(1,10), 0.0001 * ones(1,10), 0.0001 * ones(1,10)];
net.meta.trainOpts.weightDecay = 0.0001 ;
net.meta.trainOpts.batchSize = 32;
net.meta.trainOpts.numEpochs = numel(net.meta.trainOpts.learningRate) ;
net = dagnn.DagNN.loadobj(net) ;
net.layers(end-1).block.size = [1 1 2048 opts.numcls];
net.params(end-1).value = net.params(end-1).value(:,:,:,1+mod(0:opts.numcls-1,1000));
net.params(end).value = net.params(end).value(1+mod(0:opts.numcls-1,1000),:);
net.addLayer('loss', dagnn.Loss('loss', 'log'), ...
    {'prob','label'}, 'objective') ;
net.addLayer('error', dagnn.Loss('loss', 'classerror'), ...
    {'prob','label'}, 'error') ;
end

% -------------------------------------------------------------------------
function inputs = getDagNNBatch(opts, imdb, batch)
% -------------------------------------------------------------------------
imfiles = fullfile(imdb.imageDir, imdb.images.name(batch));
if ismember(imdb.images.set(batch(1)),[1,2,3])
    if opts.numGpus > 0
        images = vl_imreadjpeg(imfiles, 'Resize', [450, 450], 'Pack', 'CropLocation', 'center', 'CropSize',[1, 1], 'GPU', 'Flip', 'NumThreads', 8);
    else
        images = vl_imreadjpeg(imfiles, 'Resize', [450, 450], 'Pack', 'CropLocation', 'center', 'CropSize',[1, 1], 'Flip', 'NumThreads', 8);
    end
    images = images{1};
    avgclr = mean(mean(images(178:273,178:273,:,:),1),2);
    for idx = 1:length(imfiles)
        if rand > 0.5, images(:,:,:,idx)=flipud(images(:,:,:,idx)) ; end
        if rand > 0.5, images(:,:,:,idx)= rot90(images(:,:,:,idx)) ; end
        images(:,:,:,idx)=imrotate(images(:,:,:,idx), 11.5*(randi(8)-1), 'bilinear', 'crop'); 
    end
    labels = imdb.images.label(1,batch);
    images = bsxfun(@minus, images(114:337,114:337,:,:), avgclr);
else
    if opts.numGpus > 0
        images = vl_imreadjpeg(imfiles, 'Resize', [450, 450], 'Pack', 'CropLocation', 'center', 'CropSize',[1, 1], 'GPU', 'NumThreads', 12);
    else
        images = vl_imreadjpeg(imfiles, 'Resize', [450, 450], 'Pack', 'CropLocation', 'center', 'CropSize',[1, 1], 'NumThreads', 12);
    end
    images = images{1};
    labels = imdb.images.label(1,batch) ;
    avgclr = mean(mean(images(178:273,178:273,:,:),1),2);
    images = bsxfun(@minus, images(114:337,114:337,:,:), avgclr);
end
inputs = {'data', images, 'label', labels} ;
end


