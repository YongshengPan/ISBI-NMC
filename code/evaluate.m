function [labels, scores, predlabel] = evaluate_cell(varargin)
opts.rootPath = '../data/';
opts.trainset = [1,2,3];
opts.testset = 0;
opts.whichresnet = 50;
[opts, varargin] = vl_argparse(opts, varargin) ;
opts.expDir = fullfile(opts.rootPath, 'models/', ...
    sprintf('cell%dres-%s', opts.whichresnet, num2str(opts.trainset))) ;
% opts.modelpath =  fullfile(opts.rootPath, 'models/', sprintf('imagenet-resnet-%d-dag.mat', opts.whichresnet));
opts.modelpath =  sprintf('imagenet-resnet-%d-dag.mat', opts.whichresnet);
opts.dataDir = fullfile(opts.rootPath, '\cells');
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
net = load(fullfile(opts.expDir,'net-epoch-30.mat'));
net = dagnn.DagNN.loadobj(net.net);
        
net.move('gpu');
net.mode = 'test' ;
testID = find(ismember(imdb.images.set, opts.testset));
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
    images = cat(4, images, imrotate(images, 30, 'crop'), imrotate(images, 60, 'crop'));
    images = cat(4, images, imrotate(images, 10, 'crop'), imrotate(images, 20, 'crop'));
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
    pID(1) = single(pID(1) > 40/101);
    predlabel(idx) = 2-pID(1);
    scores(idx) = pID(2);
    if imdb.images.label(testID(idx)) ~= 2-pID(1)
        fprintf('%s %d %d.\n', imdb.images.name{testID(idx)}, 2-imdb.images.label(testID(idx)), pID(1));
    end
end

confmat = full(sparse(labels', predlabel, 1, 2, 2));
precision = diag(confmat)./sum(confmat,2);
recall = diag(confmat)./sum(confmat,1)';
f1Scores =  2*(precision.*recall)./(precision+recall);
meanF1 = mean(f1Scores);
confmat = bsxfun(@times, confmat, 1./max(sum(confmat,2),eps));
bacc = mean(diag(confmat));
fprintf('meanF1=%f, bacc=%f.\n', [meanF1, bacc]);
end