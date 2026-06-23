clear;clc   
datalist{1}='NGs';

addpath('./functions')
alpha=0.001;beta=0.1;gamma=0.001;lambda=0.001;
K=500; epilision=1e-1;
%%
%----------------
for dataset_i = 1:1
    eval(['load ./dataset/' datalist{dataset_i}]);
    fprintf('ÑµÁ·Êý¾Ý¼¯ %s \n', datalist{dataset_i});
    maxIter=100;
    kind = length(unique(Y));   
    for i =1:length(X)
        if issparse(X{i})
            X{i}=full(X{i});
        end
        X{i}=X{i}';
        X{i} = NormalizeFea(X{i},0);
    end
 
    tic
    [P,H,Z,S,obj,iter] = MSCLSP(X,alpha,beta,gamma,lambda,epilision,maxIter,K);
    time = toc;    
    Z=abs(Z)+abs(Z');

    Accmean=zeros(10,1);
    NMImean=zeros(10,1);
    ARmean = zeros(10,1);
    Fmean = zeros(10,1);
    for iters = 1:10
        [L] = SpectralClustering(Z,kind);
        L=L';
        acc = Accuracy(L',double(Y));
        [A,nmi,avgent] = compute_nmi(Y,L');
        [f,p,r] = compute_f(Y,L');
        [AR,RI,MI,HI]=RandIndex(Y,L'); 
        %
        Accmean(iters)=acc;
        NMImean(iters)=nmi;
        Fmean(iters) = f;
        ARmean(iters) = AR;
    end



end
