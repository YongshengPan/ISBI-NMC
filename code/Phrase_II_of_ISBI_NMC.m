% run(fullfile(fileparts(mfilename('fullpath')), ...
%     '..', 'matconvnet', 'matlab', 'vl_setupnn.m')) ;

% finetune_cell('ininallabel', [], 'trainset', [1,2,3], 'testset', 0, 'whichresnet', 50);
% finetune_cell('ininallabel', [], 'trainset', [1,2,3], 'testset', 0, 'whichresnet', 101);
% finetune_cell('ininallabel', [], 'trainset', [1,2,3], 'testset', 0, 'whichresnet', 152);

% [lb50_123_0, sc50_123_0, ~] = evaluate_cell('trainset', [1,2,3], 'testset', 0, 'whichresnet', 50);
rk_sc = sort(sc50_123_0);
thres = (2397+2418+2457)/(2397+2418+2457+1130+1163+1096)*1.00;
pr50_123_0 = 1+(sc50_123_0>rk_sc(round(numel(sc50_123_0)*thres)));
% pr50_123_0 = 1+(sc50_123_0>0.33);
metrics = calculate_metrics(lb50_123_0, pr50_123_0);

[fk50_123_0] = self_correct(pr50_123_0, 'trainset', [1,2,3], 'testset', 0, 'whichresnet', 50);
for epoch = 1:3
    [fk101_123_0] = self_correct(fk50_123_0, 'trainset', [1,2,3], 'testset', 0, 'whichresnet', 101);
    [fk50_123_0] = self_correct(fk101_123_0, 'trainset', [1,2,3], 'testset', 0, 'whichresnet', 50);
end

finetune_cell('ininallabel', fk50_123_0, 'trainset', [0,1,2,3], 'testset', 0, 'whichresnet', 50);
finetune_cell('ininallabel', fk101_123_0, 'trainset', [0,1,2,3], 'testset', 0, 'whichresnet', 101);
[lb50_0123_0, sc50_0123_0, pr50_0123_0] = evaluate_cell('trainset', [0,1,2,3], 'testset', 0, 'whichresnet', 50);
[lb101_0123_0, sc101_0123_0, pr101_0123_0] = evaluate_cell('trainset', [0,1,2,3], 'testset', 0, 'whichresnet', 101);
metrics = calculate_metrics(lb50_0123_0, pr50_0123_0);



function predID = self_correct(estID, varargin)
opts.rootPath = '../data/';
opts.trainset = [1,2,3];
opts.testset = 0;
opts.whichresnet = 50;
[opts, varargin] = vl_argparse(opts, varargin) ;
opts.expDir = fullfile(opts.rootPath, 'cell/', ...
    sprintf('cell%dres-%s', opts.whichresnet, num2str(opts.trainset))) ;
opts.modelpath =  sprintf('imagenet-resnet-%d-dag.mat', opts.whichresnet);
opts.dataDir = 'D:\Data\TIP(image categpry)\deep-fbanks-master\data\cell';
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.numWords = 8;
opts.numDescrsPerWord = 3000 ;
opts.train = struct('gpus', 2) ;
opts = vl_argparse(opts, varargin) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end

if exist(opts.imdbPath, 'file')
    imdb = load(opts.imdbPath) ;
else
    imdb = cell_get_database(opts);
    mkdir(opts.expDir) ;
    save(opts.imdbPath, '-struct', 'imdb');
end

net = load(fullfile(opts.expDir,'net-epoch-20.mat'));
net = dagnn.DagNN.loadobj(net.net);
idx5c = net.getVarIndex('res5c_branch2a');
for l = numel(net.layers):-1:idx5c
    layer = net.layers(l);
    net.removeLayer(layer.name);
    net.renameVar(layer.outputs{1}, layer.inputs{1}, 'quiet', true) ;
end
net.move('gpu');
net.mode = 'test' ;
testID = find(ismember(imdb.images.set, opts.testset));
feats = cell(1, numel(testID));
norml2 = @(x) bsxfun(@times, x, 1./(sqrt(sum(x.^2,1))+eps));
if exist(sprintf('feats-res%s-cv%s_%s.mat', num2str(opts.whichresnet), num2str(opts.trainset), num2str(opts.testset)),'file')
    feats = load(sprintf('feats-res%s-cv%s_%s.mat', num2str(opts.whichresnet), num2str(opts.trainset), num2str(opts.testset)));
    feats = feats.feats;
else
    for idx = 1:numel(testID)
        flnm = fullfile(imdb.imageDir, imdb.images.name{testID(idx)});
        images = single(imread(flnm));
        if size(images,1) == 600
            images = images(76:525,76:525,:);
        end
        avgclr = mean(mean(images(178:273,178:273,:,:),1),2);%./mean(mean(images(178:273,178:273,:,:)>0,1),2);
        im_resized = bsxfun(@minus, images, avgclr);
        im_resized = cat(4, im_resized, fliplr(im_resized)); im_resized = cat(4, im_resized, flipud(im_resized));
        im_resized = cat(4, im_resized, rot90(im_resized));
        im_resized =  cat(4, im_resized, imrotate(im_resized, 30, 'crop'), imrotate(im_resized, 60, 'crop'));
        im_resized =  cat(4, im_resized, imrotate(im_resized, 10, 'crop'), imrotate(im_resized, 20, 'crop'));
        %     im_resized =  cat(4, im_resized, imrotate(im_resized, 45, 'crop'));
        %     im_resized =  cat(4, im_resized, imrotate(im_resized, 22.5, 'crop'));
        %     im_resized =  cat(4, im_resized, imrotate(im_resized, 11.5, 'crop'));
        im_resized = im_resized(114:337, 114:337,:,:);
        inputs = {'data', gpuArray(im_resized)} ;
        net.eval(inputs) ;
        res = net.vars(net.getVarIndex('res5c_branch2a')).value ;
        feat = permute(gather(res), [3,1,2,4]);
        feats{idx} = norml2(reshape(feat,size(feat,1),[]));
    end
    save( sprintf('feats-res%s-cv%s_%s', num2str(opts.whichresnet), num2str(opts.trainset), num2str(opts.testset)), '-v7.3', 'feats');
end
FVENC = cell(1, numel(testID));
labels = imdb.images.label(testID);
predID = estID;
for epoch = 1:10
    [MEANS, COVARIANCES, PRIORS] = vl_gmm(vl_colsubset(cat(2,feats{:}), opts.numWords*opts.numDescrsPerWord), opts.numWords,'Initialization', 'kmeans', 'CovarianceBound', 0.0001);
    for idx = 1:numel(testID)
        FVENC{idx} = vl_fisher(feats{idx}, MEANS, COVARIANCES, PRIORS, 'Improved');
    end
    ENC = cat(2, FVENC{:});
    kernel = ENC' * ENC;
    for itr = 1:5
        fakeID = predID;
        for subi = 1:numel(fakeID)
            [simlar, subj] = maxk(kernel(subi,:), 7);
            cls = fakeID(subj)';
            if sum(cls)>=11 && predID(subi) == 1
                predID(subi) = 2;
            end
            if sum(cls)<=9 && predID(subi) == 2
                predID(subi) = 1;
            end
        end
    end
    confmat = full(sparse(labels', predID', 1, 2, 2));
    precision = diag(confmat)./sum(confmat,2);
    recall = diag(confmat)./sum(confmat,1)';
    f1Scores =  2*(precision.*recall)./(precision+recall);
    meanF1 = mean(f1Scores);
    confmat = bsxfun(@times, confmat, 1./max(sum(confmat,2),eps));
    bacc = mean(diag(confmat));
    fprintf('meanF1=%f, bacc=%f.\n', [meanF1, bacc]);
end
end

function [labels, scores, predlabel] = evaluate_cell(varargin)
opts.rootPath = '../data/';
opts.trainset = [1,2,3];
opts.testset = 0;
opts.whichresnet = 50;
[opts, varargin] = vl_argparse(opts, varargin) ;
opts.expDir = fullfile(opts.rootPath, 'cell/', ...
    sprintf('cell%dres-%s', opts.whichresnet, num2str(opts.trainset))) ;
opts.modelpath =  sprintf('imagenet-resnet-%d-dag.mat', opts.whichresnet);
opts.dataDir = 'D:\Data\TIP(image categpry)\deep-fbanks-master\data\cell';
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.whitenData = true ;
opts.contrastNormalization = true ;
opts.train = struct('gpus', 2) ;
opts = vl_argparse(opts, varargin) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end

if exist(opts.imdbPath, 'file')
    imdb = load(opts.imdbPath) ;
else
    imdb = cell_get_database(opts);
    mkdir(opts.expDir) ;
    save(opts.imdbPath, '-struct', 'imdb');
end
% gpuDevice(opts.train.gpus);
net = load(fullfile(opts.expDir,'net-epoch-30.mat'));
net = dagnn.DagNN.loadobj(net.net);
        
net.move('gpu');
net.mode = 'test' ;
testID = find(ismember(imdb.images.set, opts.testset));
rsfl = fopen('isbi_valid.predict', 'w');
predlabel = zeros(1, numel(testID));
scores = zeros(1, numel(testID));
labels = imdb.images.label(testID);
for idx = 1:numel(testID)
    flnm = fullfile(imdb.imageDir, imdb.images.name{testID(idx)});
    images = single(imread(flnm));
    if size(images,1) == 600
        images = images(76:525,76:525,:);
    end
    avgclr = mean(mean(images(178:273,178:273,:,:),1),2);

    images = cat(4, images, fliplr(images));
    images = cat(4, images, flipud(images));
    images = cat(4, images, rot90(images));
    images =  cat(4, images, imrotate(images, 30, 'crop'), imrotate(images, 60, 'crop'));
    images =  cat(4, images, imrotate(images, 10, 'crop'), imrotate(images, 20, 'crop'));
%     images =  cat(4, images, imrotate(images, 45, 'crop'));
%     images =  cat(4, images, imrotate(images, 22.5, 'crop'));
%     images =  cat(4, images, imrotate(images, 11.5, 'crop'));
    im_crop = images(114:337, 114:337,:,:);
    im_crop = bsxfun(@minus, im_crop, avgclr);
    inputs = {'data', gpuArray(im_crop)} ;
    net.eval(inputs) ;
    scr = net.vars(net.getVarIndex('prob')).value ;
    scr = squeeze(gather(scr));
    pID = mean(scr,2);
    pID(1) = single(pID(1) > 40/101); %7272/10661
    fprintf(rsfl,'%d \n', pID(1));
    predlabel(idx) = 2-pID(1);
    scores(idx) = pID(2);
    if imdb.images.label(testID(idx)) ~= 2-pID(1)
        fprintf('%s %d %d.\n', imdb.images.name{testID(idx)}, 2-imdb.images.label(testID(idx)), pID(1));
    end
end
fclose(rsfl);
zip(sprintf('ftcell%dres-cv%s', opts.whichresnet, num2str(opts.trainset)), 'isbi_valid.predict');
confmat = full(sparse(labels', predlabel, 1, 2, 2));

precision = diag(confmat)./sum(confmat,2);
recall = diag(confmat)./sum(confmat,1)';
f1Scores =  2*(precision.*recall)./(precision+recall);
meanF1 = mean(f1Scores);
confmat = bsxfun(@times, confmat, 1./max(sum(confmat,2),eps));
bacc = mean(diag(confmat));
fprintf('meanF1=%f, bacc=%f.\n', [meanF1, bacc]);
end

function [net, info] = finetune_cell(varargin)
opts.rootPath = '../data/';
opts.trainset = [1,2,3];
opts.testset = [0];
opts.whichresnet = 50;
opts.ininallabel = [];
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.expDir = fullfile(opts.rootPath, 'cell/', ...
    sprintf('cell%dres-%s', opts.whichresnet, num2str(opts.trainset))) ;
opts.modelpath =  sprintf('imagenet-resnet-%d-dag.mat', opts.whichresnet);
opts.dataDir = 'D:\Data\TIP(image categpry)\deep-fbanks-master\data\cell';
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
    avgclr = mean(mean(images(178:273,178:273,:,:),1),2);%./mean(mean(images(178:273,178:273,:,:)>0,1),2);
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
% -------------------------------------------------------------------------

function imdb = cell_get_database(opts)
opts.seed = 3;
rng(opts.seed) ;

imdb.imageDir = fullfile(opts.dataDir,'images') ;

cats = dir(imdb.imageDir) ;
cats = cats([cats.isdir] & ~ismember({cats.name}, {'.','..'})) ;
imdb.classes.name = {'all', 'hem'} ;
imdb.images.id = [] ;

for c=1:3
  all_ims = dir(fullfile(imdb.imageDir, cats(c).name, imdb.classes.name{1}, '*.bmp'));
  hem_ims = dir(fullfile(imdb.imageDir, cats(c).name, imdb.classes.name{2}, '*.bmp'));
  imNames = [cellfun(@(S) fullfile(imdb.classes.name{1}, S), {all_ims.name}, 'Uniform', 0), ...
      cellfun(@(S) fullfile(imdb.classes.name{2}, S), {hem_ims.name}, 'Uniform', 0)];
  imdb.images.name{c} = cellfun(@(S) fullfile(cats(c).name, S), imNames, 'Uniform', 0);
  imdb.images.label{c} = [1 * ones(1,numel(all_ims)), 2 * ones(1,numel(hem_ims))];
  imdb.images.set{c} = c * ones(1,numel(imNames));
  newidx = randperm(numel(imNames));
  imdb.images.name{c} = imdb.images.name{c}(1,newidx);
  imdb.images.label{c} = imdb.images.label{c}(1,newidx);
end

validlabel = readtable(fullfile(imdb.imageDir, 'isbi_valid_GT.csv'));
imNames = arrayfun(@(S) strcat('phase_2\', num2str(S),'.bmp'), 1:1867, 'Uniform', 0);
imdb.images.name{c+1} = imNames;
imdb.images.label{c+1} = 2-validlabel{:,1}';
imdb.images.set{c+1} = 0*ones(1,1867);

validlabel = readtable(fullfile(imdb.imageDir, 'isbi_test_guess.csv'));
imNames = arrayfun(@(S) strcat('phase_3\', num2str(S),'.bmp'), 1:2586, 'Uniform', 0);
imdb.images.name{c+2} = imNames;
imdb.images.label{c+2} = 2-validlabel{:,1}';
imdb.images.set{c+2} = -1*ones(1,2586);

imdb.images.name = horzcat(imdb.images.name{:}) ;
imdb.images.label = horzcat(imdb.images.label{:}) ;
imdb.images.set = horzcat(imdb.images.set{:}) ;
imdb.images.id = 1:numel(imdb.images.name) ;
imdb.meta.classes = imdb.classes.name ;
end

function metrics = calculate_metrics(labels, predlabel)
confmat = full(sparse(labels', predlabel, 1, 2, 2));
precision = diag(confmat)./sum(confmat,2);
recall = diag(confmat)./sum(confmat,1)';
f1Scores =  2*(precision.*recall)./(precision+recall);
meanF1 = mean(f1Scores);
confmat = bsxfun(@times, confmat, 1./max(sum(confmat,2),eps));
bacc = mean(diag(confmat));
metrics = [meanF1, bacc];
fprintf('WF1S=%f, BACC=%f.\n', [meanF1, bacc]);
end

