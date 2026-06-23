function [L,C] = run_kmeans(X,k)
%KMEANS Cluster multivariate data using the k-means++ algorithm.
%   [L,C] = kmeans(X,k) produces a 1-by-size(X,2) vector L with one class
%   label per column in X and a size(X,1)-by-k matrix C containing the
%   centers corresponding to each class.
%   Version: 2013-02-08
%   Authors: Laurent Sorber (Laurent.Sorber@cs.kuleuven.be)
%
%   References:
%   [1] J. B. MacQueen, "Some Methods for Classification and Analysis of 
%       MultiVariate Observations", in Proc. of the fifth Berkeley
%       Symposium on Mathematical Statistics and Probability, L. M. L. Cam
%       and J. Neyman, eds., vol. 1, UC Press, 1967, pp. 281-297.
%   [2] D. Arthur and S. Vassilvitskii, "k-means++: The Advantages of
%       Careful Seeding", Technical Report 2006-13, Stanford InfoLab, 2006.
L = [];
L1 = 0;
while length(unique(L)) ~= k
    
    % The k-means++ initialization.
    C = X(:,1+round(rand*(size(X,2)-1)));
    L = ones(1,size(X,2));
    for i = 2:k
        D = X-C(:,L);
        D = cumsum(sqrt(dot(D,D,1)));
        if D(end) == 0, C(:,i:k) = X(:,ones(1,k-i+1)); return; end
        C(:,i) = X(:,find(rand < D/D(end),1));
        [~,L] = max(bsxfun(@minus,2*real(C'*X),dot(C,C,1).'));
    end
    
    % The k-means algorithm.
    while any(L ~= L1)
        L1 = L;
        for i = 1:k, l = L==i; C(:,i) = sum(X(:,l),2)/sum(l); end
        [~,L] = max(bsxfun(@minus,2*real(C'*X),dot(C,C,1).'),[],1);
    end
    
end
% function [centroids, labels] = run_kmeans(X, k, max_iter)
% % 该函数实现Kmeans聚类
% % 输入参数：
% %                   X为输入样本集，dxN
% %                   k为聚类中心个数
% %                   max_iter为kemans聚类的最大迭代的次数
% % 输出参数：
% %                   centroids为聚类中心 dxk
% %                   labels为样本的类别标记
% 
% %% 采用K-means++算法初始化聚类中心
%   centroids = X(:,1+round(rand*(size(X,2)-1)));
%   labels = ones(1,size(X,2));
%   for i = 2:k
%         D = X-centroids(:,labels);
%         D = cumsum(sqrt(dot(D,D,1)));
%         if D(end) == 0, centroids(:,i:k) = X(:,ones(1,k-i+1)); return; end
%         centroids(:,i) = X(:,find(rand < D/D(end),1));
%         [~,labels] = max(bsxfun(@minus,2*real(centroids'*X),dot(centroids,centroids,1).'));
%   end
%   
% %% 标准Kmeans算法
%   for iter = 1:max_iter
%         for i = 1:k, l = labels==i; centroids(:,i) = sum(X(:,l),2)/sum(l); end
%         [~,labels] = max(bsxfun(@minus,2*real(centroids'*X),dot(centroids,centroids,1).'),[],1);
%   end
%   
% end