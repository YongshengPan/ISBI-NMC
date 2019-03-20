% run(fullfile(fileparts(mfilename('fullpath')), ...
%     '..', 'matconvnet', 'matlab', 'vl_setupnn.m')) ;

finetune_cell('ininallabel', [], 'trainset', [0,1,2,3], 'testset', -1, 'whichresnet', 50);
finetune_cell('ininallabel', [], 'trainset', [0,1,2,3], 'testset', -1, 'whichresnet', 101);
finetune_cell('ininallabel', [], 'trainset', [0,1,2,3], 'testset', -1, 'whichresnet', 152);

[lb50_0123_4, sc50_0123_4, pr50_0123_4] = evaluate_cell('trainset', [0,1,2,3], 'testset', -1, 'whichresnet', 50);
rk_sc = sort(sc50_0123_4);
thres = (2397+2418+2457+1219)/(2397+2418+2457+1219+1130+1163+1096+648);
pr50_0123_4 = 1+(sc50_0123_4>rk_sc(round(numel(sc50_0123_4)*thres)));
%pr50_0123_4 = 1+(sc50_0123_4>0.6);
metrics = calculate_metrics(lb50_0123_4, pr50_0123_4);

rsfl = fopen('isbi_valid.predict', 'w');
fprintf(rsfl,'%d \n', 2-pr50_0123_4);
fclose(rsfl);
zip(sprintf('BM_%dres-cv%s', 50, '0123_4'), 'isbi_valid.predict');


[fk50_0123_4] = neighborhood_correction(pr50_0123_4, 'trainset', [0,1,2,3], 'testset', -1, 'whichresnet', 50);
for epoch = 1:5
    [fk101_0123_4] = neighborhood_correction(fk50_0123_4, 'trainset', [0,1,2,3], 'testset', -1, 'whichresnet', 101);
    [fk50_0123_4] = neighborhood_correction(fk101_0123_4, 'trainset', [0,1,2,3], 'testset', -1, 'whichresnet', 50);
end

rsfl = fopen('isbi_valid.predict', 'w');
fprintf(rsfl,'%d \n', 2-fk50_0123_4);
fclose(rsfl);
zip(sprintf('NCA_%dres-cv%s', 50, '0123_4'), 'isbi_valid.predict');

rsfl = fopen('isbi_valid.predict', 'w');
fprintf(rsfl,'%d \n', 2-fk101_0123_4);
fclose(rsfl);
zip(sprintf('NCA_%dres-cv%s', 101, '0123_4'), 'isbi_valid.predict');


%%%if you want to run the follow code, you should crop all images to size of 450x450
%
% finetune_cell('ininallabel', fk50_0123_4, 'trainset', [0,1,2,3,-1], 'testset', -1, 'whichresnet', 50);
% finetune_cell('ininallabel', fk101_0123_4, 'trainset', [0,1,2,3,-1], 'testset', -1, 'whichresnet', 101);
% [lb50_01234_4, sc50_01234_4, pr50_01234_4] = evaluate_cell('trainset', [0,1,2,3,-1], 'testset', -1, 'whichresnet', 50);
% [lb101_01234_4, sc101_01234_4, pr101_01234_4] = evaluate_cell('trainset', [0,1,2,3,-1], 'testset', -1, 'whichresnet', 101);
% metrics = calculate_metrics(lb50_01234_4, pr50_01234_4);
% metrics = calculate_metrics(lb101_01234_4, pr101_01234_4);


