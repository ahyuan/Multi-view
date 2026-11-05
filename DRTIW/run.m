clc;  close all; clear all;
currentFolder = pwd;
addpath(genpath(currentFolder));
addpath('functions/');
resultdir = 'res/';
datadir = 'datasets/';  
if(~exist('res','file'))
    mkdir('res');
    addpath(genpath('res/'));
end
dataname = {'MSRCV1'};

numdata = length(dataname);


retry = 10; 
maxIter = 100;
sigma = 2;
ALPHA = [0.001,0.01,0.1,1,10,100,1000];
BETA = [0.001,0.01,0.1,1,10,100,1000];


for cdata = 1:numdata  
    %% read dataset
    idata = cdata;
    dataf = [datadir, cell2mat(dataname(idata))]

	data = load(dataf);

	X= data.X;
	Y_real = data.Y;


clu_num = max(Y_real);
view_num = length(X);
for v = 1:view_num
	X{v} = X{v}';
    X{v} = NormalizeFea(X{v},0);
end



[~,n] = size(X{1});


filepath = strcat(resultdir,char(dataname(idata)),'_Result.csv');
fid = fopen(filepath, 'w');
if fid == -1
	error('No such dir,Create Results dir');
end
fprintf(fid, 'alpha,beta,gamma,sigma,ACCmean,ACCstd,NMImean,NMIstd,FscoreMean,FscoreSTD,ARImean,ARIstd\n');
	
count = 0;
for a = 1:length(ALPHA)
	for j = 1:length(BETA)
		% for z = 1:length(GAMMA)
            P_list = zeros(retry,0);
            ACC_list = zeros(retry,0);
            NMI_list = zeros(retry,0);
            F_list = zeros(retry,0);
            AR_list = zeros(retry,0);
            RI_list = zeros(retry,0);
            R_list = zeros(retry,0);
            [B,B_v,H_v,H,w,Z,obj] = MVC(X,ALPHA(a),BETA(j),clu_num,maxIter);
            for t = 1:retry								
				Y_pre = kmeans(H', clu_num, 'MaxIter', 200, 'Replicates', 20);				
				acc = Accuracy(Y_pre,Y_real);
				[~,nmi,~] = compute_nmi(Y_real,Y_pre);
				[f,p,r] = compute_f(Y_real,Y_pre);
				[AR,RI,MI,HI]=RandIndex(Y_real,Y_pre);                 
                ACC_list(t) = acc;
                NMI_list(t) = nmi;
                F_list(t) = f;
                AR_list(t) = AR;				
            end
            ACC_mean = mean(ACC_list);
            ACC_std = std(ACC_list);
            NMI_mean = mean(NMI_list);
            NMI_std = std(NMI_list);
            F_mean = mean(F_list);
            F_std = std(F_list);
            AR_mean = mean(AR_list);
            AR_std = std(AR_list);
			fprintf(fid, '%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n', ...
                ALPHA(a),BETA(j),0,0,ACC_mean,ACC_std,NMI_mean,NMI_std,F_mean,F_std,AR_mean,AR_std);
            count = count+1;
			disp(count);
	end
end
end

	