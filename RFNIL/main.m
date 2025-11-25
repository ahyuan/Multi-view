clear
% clc

%%RI_fixed


path = "\datasets\NGs.mat"



%%ORL sigma2 80
%%BBCsport 80.


GAMMA = 0.1;
BETA = 0.1;
LAMBDA = 0.1;
exl_colmn_num = 20;
% BETA = [0.001, 0.01, 0.1, 1, 10, 100];
% GAMMA = [0.001, 0.01, 0.1, 1, 10, 100];
% LAMBDA = [0.001, 0.01, 0.1, 1, 10, 100];
sigma = 3;
% maxIter = 7;
% runtimes = 10;
fea_rate = 10:10:300;
runtimes = 10; % run-times of clustering method



data = load(path);
X= data.X;
Y_real = data.Y;

% X = data.data;
% view_num = length(X);
% for v = 1:view_num
%    X{v} = X{v}';
% end
% Y_real = data.truelabel;
% Y_real = Y_real{1}';


%类下标最大的一个就是类别总数
clu_num = max(Y_real);
c = length(unique(Y_real));

view_num = length(X);
%目前的数据集样本都是行排的，所以还需要变为排
for v = 1:view_num
	if issparse(X{v})
            X{v}=full(X{v});
	end
	X{v} = X{v}';
    X{v} = NormalizeFea(X{v},0);
    % X{v} = normalize(X{v})
    %%%%%%matlab自带的归一化程序怎么样
end
% 
% view_num = 3;
% X = cell(1,view_num);
% X{1} = NormalizeFea(data.X1,0);
% X{2} = NormalizeFea(data.X2,0);
% X{3} = NormalizeFea(data.X3,0);
% Y_real = double(data.gt);
% clu_num = max(Y_real);


[~,n] = size(X{1});
options = struct();
options.v = view_num; options.c=clu_num; options.n=n;

S_v = cell(view_num,1);
for v = 1:view_num
	data_v = X{v}';%%pdist要求行排列
	Dis = squareform(pdist(data_v,'squaredeuclidean'));
	Dis = -1.*Dis./sigma;
	S_temp = exp(Dis);
	rowSums = sum(S_temp,2);
	rowSum_v = repmat(rowSums,1,size(S_temp,2));
	S_temp = S_temp./rowSum_v;
	S_v{v} = (S_temp+S_temp')/2;
	
end
clear data_v Dis S_temp rowSums rowSum_v;

count = 0;
RESULT = zeros(length(BETA)*length(GAMMA)*length(LAMBDA),exl_colmn_num);
for a = 1:length(BETA)
	for j = 1:length(GAMMA)
		for z = 1:length(LAMBDA)
		
			tic
			[P, Y_hat, Y, S, alpha, obj] = RFNIL(X, S_v, BETA(a), GAMMA(j),LAMBDA(z), options);
			time = toc;
			
			XX = [];
			W = [];
			for v = 1:view_num
				W=[W;sum(P{v}.*P{v},2)];  % 对行求2范数
				XX=[XX;X{v}];
			end
			[~,index] = sort(W,'descend');
			for i_fea = fea_rate
				new_fea = XX(index(1:floor(i_fea)),:);
				ACC_list = zeros(runtimes,0);
				NMI_list = zeros(runtimes,0);
				Purity_list = zeros(runtimes,0);
				F_list = zeros(runtimes,0);
				Precision_list = zeros(runtimes,0);
				R_list = zeros(runtimes,0);
				AR_list = zeros(runtimes,0);
				for i = 1:runtimes
					Label = litekmeans(new_fea', c,'Replicates',5); %label by k-means
					result = Clustering8Measure(Y_real, Label);
					ACC_list(i)=result(1);
					NMI_list(i)=result(2);
					Purity_list(i)=result(3);
					F_list(i)=result(4);
					Precision_list(i)=result(5);
					R_list(i)=result(6);
					AR_list(i)=result(7);
				end
				ACCmean=mean(ACC_list);
				NMImean=mean(NMI_list);
				ACCstd=std(ACC_list);
				NMIstd=std(NMI_list);
				Pumean = mean(Purity_list);
				Pustd = std(Purity_list);
				Fmean = mean(F_list);
				Fstd = std(F_list);
				Premean = mean(Precision_list);
				Prestd = std(Precision_list);
				Rmean = mean(R_list);
				Rstd = std(R_list);
				ARmean = mean(AR_list);
				ARstd = std(AR_list);
				% fprintf(fid, '%f,%f,%f,%f,%f,%f,%f,%f\n', beta,gamma,lambda,i_fea,ACCmean,ACCstd,NMImean,NMIstd);
				% disp(['ACC  ',num2str(ACCmean)]);
				% disp(['NMI  ',num2str(NMImean)]);
                count = count+1;
			disp(count);
            RESULT(count,:) = [BETA(a),GAMMA(j),LAMBDA(z),sigma,i_fea,ACCmean,ACCstd,NMImean,NMIstd,Fmean,Fstd,ARmean,ARstd,Pumean,Pustd,Rmean,Rstd,Premean,Prestd,time];
            disp(RESULT(count,:))
            end
            
            
        end
	end
end
% write(RESULT,'YaleA_3view.xlsx');
% xlswrite('Caltech101-all_fea.xlsx',RESULT);






