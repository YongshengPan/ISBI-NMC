function crop_cell(varargin)
opts.rootPath = '../data/';
opts.trainset = [0,1,2,3];
opts.testset = -1;
opts.whichresnet = 50;
opts.expDir = fullfile(opts.rootPath, 'models/', ...
    sprintf('cell%dres-%s', opts.whichresnet, num2str(opts.trainset))) ;
[opts, varargin] = vl_argparse(opts, varargin) ;
opts.dataDir = fullfile(opts.rootPath, '\cells');
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');

opts = vl_argparse(opts, varargin) ;

if exist(opts.imdbPath, 'file')
    imdb = load(opts.imdbPath) ;
else
    imdb = cell_get_database(opts);
    mkdir(opts.expDir) ;
    save(opts.imdbPath, '-struct', 'imdb');
end

testID = find(ismember(imdb.images.set, opts.testset));
for idx = 1:numel(testID)
    flnm = fullfile(imdb.imageDir, imdb.images.name{testID(idx)});
    images = imread(flnm);
    if size(images,1) == 600
        images = images(76:525,76:525,:);
        imwrite(images, flnm);
    end
end
end