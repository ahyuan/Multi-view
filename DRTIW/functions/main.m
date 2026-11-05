clear
% clc

%%RI_fixed

% path = "F:\end2endMVC\data\multi-view-dataset-master\20newsgroups.mat"
% path = "F:/end2endMVC/data/Wiki_fea.mat"
path = "D:\原D\毕设-基于低秩半非负矩阵分解的多视图数据聚类算法研究\data/bbc.mat"
% path = "F:/end2endMVC/data/WikipediaArticles.mat"
% path = "F:/end2endMVC/data/WebKB_2views.mat"
% path = "F:/end2endMVC/data/Wiki_fea.mat"
% path = "F:/end2endMVC/data/WebKB_cor2views_cornell.mat"
% path = "F:/end2endMVC/data/uci-digit.mat"


%%ORL sigma2 80
%%BBCsport 80.


% alpha = 0.05;
% beta = 0.01;
% gamma = 0.01;
exl_colmn_num = 17;
% ALPHA = [0.0001,0.001,0.01,0.1,1,10,100,1000];
% BETA = [0.0001,0.001,0.01,0.1,1,10,100,1000];
% GAMMA = [0.0001,0.001,0.01,0.1,1,10,100,1000];
ALPHA = [0.01];
BETA = [1];
GAMMA = [0.1];
sigma = 3;
maxIter = 10;
retry = 10;

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
view_num = length(X);
%目前的数据集样本都是行排的，所以还需要变为排
for v = 1:view_num
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
count = 0;
RESULT = zeros(length(ALPHA)*length(BETA)*length(GAMMA),exl_colmn_num);
for a = 1:length(ALPHA)
	for j = 1:length(BETA)
		for z = 1:length(GAMMA)
            P_list = zeros(retry,0);
            ACC_list = zeros(retry,0);
            NMI_list = zeros(retry,0);
            F_list = zeros(retry,0);
            AR_list = zeros(retry,0);
            RI_list = zeros(retry,0);
            R_list = zeros(retry,0);
            for t = 1:retry
				[P,Y_p,Y,mu,R,S,obj] = AS3MVC(X,ALPHA(a),BETA(j),GAMMA(z),sigma,clu_num,maxIter);
				Y_pre = zeros(n,1);
				for i = 1:n
					[~,k] = max(Y(i,:));
					Y_pre(i) = k;
				end

				acc = Accuracy(Y_pre,Y_real);
				[~,nmi,~] = compute_nmi(Y_real,Y_pre);
				[f,p,r] = compute_f(Y_real,Y_pre);
				[AR,RI,MI,HI]=RandIndex(Y_real,Y_pre); 
                
                P_list(t) = p;
                ACC_list(t) = acc;
                NMI_list(t) = nmi;
                F_list(t) = f;
                AR_list(t) = AR;
                RI_list(t) = RI;
                R_list(t) = r;
				% disp(num2str(acc));
				% disp(num2str(nmi));
				% disp(num2str(f));
				% disp(num2str(AR));
				
            end
            ACC_mean = mean(ACC_list);
            ACC_std = std(ACC_list);
            NMI_mean = mean(NMI_list);
            NMI_std = std(NMI_list);
            F_mean = mean(F_list);
            F_std = std(F_list);
            AR_mean = mean(AR_list);
            AR_std = std(AR_list);
            RI_mean = mean(RI_list);
            RI_std = std(RI_list);
            p_mean = mean(P_list);
            p_std = std(P_list);
            R_mean = mean(R_list);
            R_std = std(R_list);
            count = count+1;
			disp(count);
            RESULT(count,:) = [ALPHA(a),BETA(j),GAMMA(z),ACC_mean,ACC_std,NMI_mean,NMI_std,F_mean,F_std,AR_mean,AR_std,RI_mean,RI_std,p_mean,p_std,R_mean,R_std];
            disp(RESULT(count,:))
        end
	end
end
xlswrite('orl_2515k_spec.xlsx',RESULT);






