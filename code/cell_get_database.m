function imdb = cell_get_database(opts)
opts.seed = 3;
rng(opts.seed) ;

imdb.imageDir = fullfile(opts.dataDir,'') ;
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


