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
