function [objectives, score, index] = AGFD(X, beta, r, tau, NITER, NMF_k, vN)
%X:输入的数据矩阵
%beta,r,epsilon:
%NITER:最大迭代次数
%NMF_k:矩阵分解后的维度
%vN:视图数量
%Alpha:一致性权重系数
% Alpha = abs(rand(vN, 1));

Alpha = ones(vN, 1) / vN;
[n, ~] = size(X{1});
U_div = cell(1, vN);
U_con = abs(rand(n, NMF_k))* 0.1;
%     X1 = X{1};
%     U_con = pca(X, NMF_k).scores;
V = cell(1, vN);
SS = cell(1, vN);
LL = cell(1, vN);
NorX = [];

for i = 1:length(X)
    NorX = [NorX, X{i}];
end

options = [];
options.NeighborMode = 'KNN';
options.k = 5;
options.WeightMode = 'HeatKernel';
options.t = 5;

S = constructW(NorX,options);
S = NormalizeFea(S,1);
L = diag(sum(S, 1)) - S;

S = zeros(n, n);
%% 初始化
tic;
disp('初始化时间：');
for vIndex = 1:vN
    [n, d] = size(X{vIndex});
    U_div{vIndex}= (rand(n, NMF_k));
    
    V{vIndex} = abs(rand(d, NMF_k));
    SS{vIndex} = constructW(X{vIndex}, options);
    SS{vIndex} = NormalizeFea(SS{vIndex}, 1);
    LL{vIndex} = diag(sum(SS{vIndex}, 1)) - SS{vIndex};
end

toc;

diff = 1; iteration = 1;
objectives = zeros(NITER, 1);
D_kl = zeros(vN, 1);

T = cell(vN, 1);

for i = 1:vN
    T{i} = zeros(n, n);
end
T1_iter = cell(1, NITER);
T2_iter = cell(1, NITER);
while (iteration <= NITER) %diff > 0.1 &&
    tic;
    disp('alpha更新:');
    %% 更新Alpha
    temp = zeros(vN, 1);
    
    for vIndex = 1:vN
        temp(vIndex) = norm((X{vIndex} - (U_con + U_div{vIndex}) * V{vIndex}'), 'fro') ^ 2 - sum(sqrt(sum(V{vIndex} .^ 2, 2))) + norm(U_div{vIndex}, 1) + D_kl(vIndex);
        temp(vIndex) = abs(temp(vIndex));
    end
    
    Alpha = temp .^ (1 / (1 - r)) / sum(temp .^ (1 / (1 - r)));
    toc;
    
    %% 更新V
    tic;
    disp('V更新:');
    for vIndex = 1:vN
        D_weight = diag(0.5 ./ sqrt(sum(V{vIndex} .* V{vIndex}, 2) + eps));
        V_up = 2 * Alpha(vIndex) ^ r * X{vIndex}' * (U_con + U_div{vIndex});
        V_down = 2 * Alpha(vIndex) ^ r * V{vIndex} * (U_con' + U_div{vIndex}') * (U_con + U_div{vIndex}) + (1 - Alpha(vIndex) ^ r) * D_weight * V{vIndex};
        V{vIndex} = V{vIndex} .* (V_up ./ V_down+eps);
        V{vIndex} = max(V{vIndex}, 0);
    end
    toc;
    
    %% 更新Udiv
    
    for vIndex = 1:vN
        [x, error] = updateUdiv(V{vIndex}, X{vIndex}, U_con, Alpha(vIndex) ^ r);
        U_div{vIndex} = reshape(x, [n, NMF_k]);
    end
    
    %% 更新Ucon
    
    tic;
    disp('Ucon更新：');
    T1 = zeros(n, NMF_k);
    T2 = zeros(n, NMF_k);
    
    for vIndex = 1:vN
        T1 = T1 + Alpha(vIndex) ^ r .* X{vIndex} * V{vIndex};
        T2 = T2 + Alpha(vIndex) ^ r .* ((U_con + U_div{vIndex}) * V{vIndex}' * V{vIndex} + LL{vIndex}*U_con);
    end
    T1_iter{iteration} = T1;
    T2_iter{iteration} = T2;
    
    U_con = U_con .* (T1 + 2 * tau * U_con) ./ (T2+beta * L * U_con+2 * tau *  U_con+eps);    
    U_con = max(U_con, 0);
    toc;
      
    tic;
    disp('S的更新：');
    %% 更新S
    for i = 1:n
        for j = 1:n
            aij = 0.5 * beta * norm(U_con(i, :) - U_con(j, :), 2);
            t_up = 0;
            t_down = 0;
            for vIndex = 1:vN
                t_up = t_up + Alpha(vIndex) ^ r * (log(SS{vIndex}(i, j) + eps) - 1);
                t_down = t_down + Alpha(vIndex) ^ r;
            end
            S(i, j) = exp((-aij + t_up) / t_down);
        end
        S(i, :) = S(i, :) / sum(S(i, :));
    end
    S = (S + S') / 2;
    S = NormalizeFea(S, 1);
    L = diag(sum(S, 1)) - S;
    
    for vIndex = 1:vN
        D_kl(vIndex) = sum(sum(S .* log(((S + eps) ./ (SS{vIndex} + eps)))));
    end
    
    toc;
    %% objective function value
    sum_term1 = 0;
    sum_term2 = 0;
    sum_term3 = 0;
    sum_term4 = 0;
    sum_term5 = 0;
    sum_term6 = 0;
    sum_term4_values = zeros(vN, 1);
    for vIndex = 1:vN
        sum_term1 = sum_term1 + Alpha(vIndex) ^ r * norm((X{vIndex} - (U_con + U_div{vIndex}) * V{vIndex}'), 'fro') ^ 2;
        sum_term2 = sum_term2 + (1 - Alpha(vIndex) ^ r) * sum(sqrt(sum(V{vIndex} .^ 2, 2)));
        sum_term3 = sum_term3 + Alpha(vIndex) ^ r * D_kl(vIndex);
        sum_term4 = sum_term4 + Alpha(vIndex) ^ r * trace(U_con' * L * U_con);
        sum_term6 = Alpha(vIndex) ^ r * norm(U_div{vIndex}, 1);
    end
    
    sum_term5 = beta * trace(U_con' * L * U_con);
    Tempobj = sum_term1+ sum_term2+sum_term3+sum_term4+sum_term5+sum_term6;
    objectives(iteration) = Tempobj;
    % disp(iteration);
    if iteration > 1
        diff = abs(objectives(iteration - 1) - objectives(iteration));
    end
    
    disp(['迭代次数：' num2str(iteration) '目标函数值：' num2str(objectives(iteration))]);
    iteration = iteration + 1;
end

Vcon = [];

for vIndex = 1:vN
    Vcon = [Vcon; V{vIndex}];
end

score = sum(Vcon .* Vcon, 2);
[~, index] = sort(score, 'descend');
end
