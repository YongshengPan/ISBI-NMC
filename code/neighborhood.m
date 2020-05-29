function predID = neighborhood_correction(estID, varargin)
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
opts.numWords = 8 ;
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