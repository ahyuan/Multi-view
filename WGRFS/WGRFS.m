function [ranking, XX] = WGRFS(X, alpha, beta, c)

    %% 参数设置，变量初始化
    V = size(X, 2); % 视图数量
    n = size(X{1, 1}, 1); % 样本数量

    % 最大最小归一化
    XX = []; % 特征拼接
    for v = 1:V
        X{1, v} = (X{1, v} - repmat(min(X{1, v}), n, 1)) ./ repmat(max(X{1, v}) - min(X{1, v}), n, 1);
        X{1, v}(isnan(X{1, v})) = 1; 
        XX = [XX X{1, v}];
    end

    % 初始化相似性矩阵 S
    S = cell(1, V);
    for v = 1:V
        sigma = optSigma(X{1,v})^2; %计算的是所有样本对之间的欧几里得距离的中值，随后平方得到参数 sigma
        S{1,v} = constructW(X{1,v}, struct('k',5, 'WeightMode', 'HeatKernel', 't', sigma));
        S{1,v} = V * S{1,v} ./ repmat( sum(S{1,v},2 ) , 1 , size(S{1,v},1)); %将每一行的和扩展到整个矩阵，然后进行按行归一化
    end

    % 初始化 P, D, F, W, theta
    P = cell(1, V); %投影矩阵P
    D = cell(1, V); %对角矩阵D，由P构造
    m = cell(1, V);
    F = randn(n, c);
    W = ones(V, V);
    W(1:V+1:end) = 0; % 将W对角元素设置为 0
    W = W ./ V; % 将W非对角元素除以 V
    theta = ones(1, V) / V;
    for v = 1:V
        m{1, v} = size(X{1, v}, 2); % 特征维度
        X{1, v} = X{1, v}'; % 每个视图的样本转置,现在X^v的维度是m^v*n
        P{1, v} = eye(m{1, v}, c); % 初始化 P
        D{1, v} = eye(m{1, v}); % 初始化 D
    end

    %% 优化过程
    MAXITER = 20; % 最大迭代次数
    res = zeros(MAXITER, 1); % 每次迭代的目标函数值

    % 计算初始目标函数值
    res_one = 0;
    res_two = 0;
    for v = 1:V
        for i = 1:n
            for j = 1:n
                res_one = res_one + theta(v)^2 * sum((F(i, :) - (X{1, v}(:, j)' * P{1, v})).^2) * S{1, v}(i, j);
            end
        end
        res_one = res_one + alpha * norm_21(P{1, v});
    end   
    %计算W的相关项
    WTerm = 0;
    for i = 1:V
        for j = 1:V
            WTerm = WTerm + W(i, j) * (S{1, i}(:)' * S{1, j}(:));
        end
    end    
    res_two = res_two + beta * WTerm + norm(W,'fro')^2;
    res_old = res_one + res_two; 
    res(1) = res_old;

    for iter = 1:MAXITER   
        % 更新 P 和 D  
        for v = 1:V
            temp_P = (theta(v)^2 * X{1, v} * X{1, v}')+(alpha * D{1, v});
            P{1, v} = temp_P \ (theta(v)^2 * X{1, v} * S{1, v}' * F);
            tempD = 0.5 * (sqrt(sum(P{1, v}.^2, 2) + eps)).^(-1);
            D{1, v} = diag(tempD);
        end

        % 更新 F
        for v = 1:V
            SVD = S{1, v} * X{1, v}' * P{1, v};
            [U_F, ~, V_F] = svd(SVD, 'econ');
            F = U_F * V_F';
        end
        
        % 更新 theta
        b = zeros(V, 1);
        for v = 1:V
            term = 0;
            for i = 1:n
                for j = 1:n
                    term = term + norm(F(i,:) - X{1,v}(:,j)'*P{1,v}, 2)^2;
                    b(v) = b(v) + S{1, v}(i, j) * term;
                end
            end
            inv_b = 1 ./ (b+eps); 
            theta = inv_b / sum(inv_b);
        end        

        % 更新 S
        for v = 1:V
            % 计算 T 和 H
            T = zeros(n, n);
            for i = 1:n
                for j = 1:n
                    T(i, j) = theta(v)^2 * sum((F(i, :) - (X{1, v}(:, j)' * P{1, v})).^2);
                end
            end

            H = zeros(n, n);
            for k = 1:V
                if k ~= v
                    H = H + S{1, k} * W(k, v);
                end
            end

            % 计算 J
            J = H - (1 / (4 * beta)) * T;

            % 更新 S^{(v)}
            for i = 1:n
                for j = 1:n
                    if i ~= j
                        S{1, v}(i, j) = max(J(i, j) + (1 - sum(J(i, [1:i-1, i+1:end]))) / (n - 1), 0);
                    else
                        S{1, v}(i, j) = 0;
                    end
                end
            end
        end
        
        % 更新 W
        A = zeros(n^2, V);
        for v = 1:V
            A(:, v) = S{v}(:);
        end
        for v = 1:V
            % 计算 beta * A' * A
            beta_A_trans_A = beta * (A' * A);

            % 计算 beta * A' * A(:, v)
            beta_A_trans_A_v = beta * A' * A(:, v);
    
            % 计算 M = (beta * A' * A + eye(V))^{-1}
            M = (beta_A_trans_A + eye(V)) \ eye(V);

            % Calculate delta
            delta = 2 * (ones(1, V) * M * beta_A_trans_A_v - 1) / (ones(1, V) * M * ones(V, 1));

            % Update W_v
            W(:, v) = M * (beta_A_trans_A_v - (delta / 2) * ones(V, 1));
    
            % Ensure the constraints are satisfied
            W(v, v) = 0; % W_vv = 0
            W(W(:, v) < 0, v) = 0; % Ensure 0 <= W_kv
            W(W(:, v) > 1, v) = 1; % Ensure W_kv <= 1
        end

        % 计算新的目标函数值
        res_one = 0;
        res_two = 0;
        for v = 1:V
            for i = 1:n
                for j = 1:n
                    res_one = res_one + theta(v)^2 * sum((F(i, :) - (X{1, v}(:, j)' * P{1, v})).^2) * S{1, v}(i, j);
                end
            end
            res_one = res_one + alpha * norm_21(P{1, v});
        end     
        %计算W的相关项
        WTerm = 0;
        for i = 1:V
            for j = 1:V
                WTerm = WTerm + W(i, j) * (S{1, i}(:)' * S{1, j}(:));
            end
        end    
        res_two = res_two + beta * WTerm +  norm(W,'fro')^2;
        res_new = res_one + res_two; 
        res(iter + 1) = res_new;

        % 判断是否收敛
        fprintf('Iter = %d; Objective value = %f\n', iter, res_new)
        diff = res_old - res_new;
        if (iter > 1 && abs(diff) / res_old < 1e-4) || (iter > 1 && abs(diff) < 1e-4)
            break
        else
            res_old = res_new;
        end
    end

    % 计算特征排序
    PP = [];
    for v = 1:V
        PP = [PP; P{1, v}];
    end
    [~, ranking] = sort(sum(PP .* PP, 2), 'descend');

end

function sigma = optSigma(X)
%input£ºX: row-sample  column-feature
%output:sigma
N = size(X,1); %sample number
dist = EuDist2(X,X);   
dist = reshape(dist,1,N*N); 
sigma = median(dist); 
end

function norm_value = norm_21(P)
    % 计算 W 的 2,1 范数
    norm_value = sum(sqrt(sum(P.^2, 2)));
end