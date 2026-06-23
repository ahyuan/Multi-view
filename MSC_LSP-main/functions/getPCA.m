function [H] = getPCA(X,K)
% 输入X n*m的，n是数据量，m是特征
% X = zscore(X);
% R = corr(X);
% [V,D] = eig(R);
% lam = diag(D);
% [lam_sort, index] = sort(lam,'descend');
% V_sort = V(:,index);
% % 找前K个
% index1 = K;
% H = X*V_sort;
% H = H(:,1:index1);
% H = H';
[coeff,~,~] = pca(X);
H = X*coeff(:,1:K);
H = H';