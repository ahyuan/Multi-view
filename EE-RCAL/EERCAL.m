function [A,W,Z,D,G,F,iter,obj,alpha,gamma] = EERCAL(X,Y,lambda,d,numanchor)
%初始化
maxIter = 12 ;
m = numanchor;
numclass = length(unique(Y));
numview = length(X);
numsample = size(Y,1);
W = cell(numview,1);
W_f = cell(numview,1);
A = zeros(d,m);
Z = zeros(m,numsample);
D = cell(numview,1);
D_sd = cell(numview,1);
alpha = ones(1,numview)/numview;
gamma = ones(1,numview)/numview;
for i = 1:numview
   di = size(X{i},1);
   W{i} = zeros(di,d);
   D{i} = eye(m,numsample);
   try
       X{i} = mapstd(X{i}',0,1);
   catch
       tmp = X{i}';
       tmp = (tmp - mean(tmp,1)) ./ max(1e-12, std(tmp,0,1));
       X{i} = tmp;
   end
end
Z(:,1:m) = eye(m);
Z_fp = ones(m,numsample)./m;
G = eye(m,numclass);
F = eye(numclass,numsample);
M_W = cell(numview,1);
for i=1:numview
    M_W{i} = zeros(d,numsample);
end

eps_aux = 1e-8;
flag = 1;
iter = 0;
f_F = zeros(numclass,numsample);

for p = 1:numview
    dp = size(X{p},1);
    % 初始化辅助正交基 
    Rnd = randn(dp,d);
    if dp >= d
        [Qf,~] = qr(Rnd,0);
        W_f{p} = Qf(:,1:d);
    else
        [Qf,~] = qr([Rnd; randn(d-dp,d)],0);
        W_f{p} = Qf(1:dp,1:d);
    end
 
    D_sd{p} = ones(m,numsample) ./ m;
end

% 其他随机初始化
[Ua_f,~,Va_f] = svd(randn(d,m),'econ');
f_A = Ua_f * Va_f';
[Ug_f,~] = qr(randn(m,numclass),0); f_G = Ug_f(:,1:numclass);
idx_rand = randi(numclass,numsample,1);
for i = 1:numsample
    f_F(idx_rand(i), i) = 1;
end

while flag
    iter = iter + 1;
    AZ = A * Z;
    Delta = 1e-6;
    for i = 1:numview
        M_W{i} = A * (Z + D{i});
    end

    %  W{iv} 
    parfor iv = 1:numview
        %   SVD 更新正交矩阵
        Cmat = X{iv} * AZ';                 % 尺寸: (dim_iv x d) * (d x m)' => dim_iv x m
        [UJP,~,VJP] = svd(Cmat,'econ');
        W{iv} = UJP * VJP';                 % 保持为正交矩阵近似

        % 2) 更新辅助正交矩阵 
        dp = size(X{iv},1);
        % W_f 已在外部初始化，保留并用其当前值
        sq = sum(W_f{iv}.^2, 2);            % 每行平方和
        Qdiag = 1 ./ sqrt(sq + 1e-8);       % 每行的 reweight 倒数
        eta = max(Qdiag);                   % 用最大元素构造缩放
        Qhat = eta * eye(dp) - diag(Qdiag);
        % ：f_A (d x m), (Z_fp + D_sd{iv}) (m x samples) -> d x samples
        M_p_all = f_A * (Z_fp + D_sd{iv});  % d x samples
        % 生成 Jp（d x d 矩阵或 d x ?），将数据项合并为与 W_f 兼容的尺寸
        term1 = Qhat * W_f{iv};             % dp x d
        term2 = 2 * (X{iv} * M_p_all');     % dp x d  (因为 X{iv}: dp x samples, M_p_all': samples x d)
        Jp = term1 + term2;
        Jp(~isfinite(Jp)) = 0;
        [Ufaux,~,Vfaux] = svd(Jp,'econ');
        W_f{iv} = Ufaux * Vfaux';
    end
    % 并行更新

    % Update A
    sumAlpha = 0;
    part1 = zeros(d,m);
    Jmat_f = zeros(d,m);
    for p = 1:numview
        Mtmp = Z + D{p};
        tmp = W{p}' * X{p} * Mtmp';
        tmp(~isfinite(tmp)) = 0;
        Jmat_f = Jmat_f + 2 * (alpha(p)^2) * tmp;
    end
    [Ua_f,~,Va_f] = svd(Jmat_f,'econ');
    if ~isempty(Ua_f) && ~isempty(Va_f)
        f_A = Ua_f * Va_f';
    end
    for ia = 1:numview
        al2 = alpha(ia)^2;
        sumAlpha = sumAlpha + al2;
        part1 = part1 + al2 * W{ia}' * X{ia} * Z';
    end
    [JA1,~,JA2] = svd(part1,'econ');
    A = JA1 * JA2';
    Jmat_f = zeros(d,m);
    for p = 1:numview
        Mtmp = Z + D{p};
        tmp = W{p}' * X{p} * Mtmp';
        tmp(~isfinite(tmp)) = 0;
        Jmat_f = Jmat_f + 2 * (alpha(p)^2) * tmp;
    end
    [Ua_f,~,Va_f] = svd(Jmat_f,'econ');
    if ~isempty(Ua_f) && ~isempty(Va_f)
        f_A = Ua_f * Va_f';
    end

    % Update Z
    HZ = 2*sumAlpha*eye(m) + 2*lambda*eye(m);
    HZ = (HZ + HZ')/2;
    options = optimset('Algorithm','interior-point-convex','Display','off');
    parfor ji = 1:numsample
        ff = zeros(m,1);
        kz = G * F(:,ji);
        for j = 1:numview
            Cmat = W{j} * A;
            ff = ff - 2 * (Cmat' * X{j}(:,ji)) - 2 * lambda * kz;
        end
        Z(:,ji) = quadprog(HZ, ff, [], [], ones(1,m), 1, zeros(m,1), ones(m,1), [], options);
    end

    % Update D 
    QD = cell(1,numview);
    for i = 1:numview
        dv = size(D{i},1);
        x = zeros(dv,1);
        for j = 1:dv
            x(j) = 0.5 * norm(D{i}(j,:),2) + Delta;
        end
        QD{i} = diag(x);
    end
    for p = 1:numview
        gp = max(gamma(p), 1e-12);
        denomD = 1 + 4 * ( (gp/gp)^2 ); 
        T = X{p} - 0.5 * W_f{p} * f_A * Z_fp;
        Wa = W_f{p} * f_A;
        Qall = (T' * Wa) * (2 / denomD);
        Qall_t = Qall';
        for ii = 1:numsample
            qi = Qall_t(:,ii);
            qi(~isfinite(qi)) = 0;
            D_sd{p}(:,ii) = proj_simplex(qi);
        end
    end

    % Update G
    Jg = Z * F';
    [Ug,~,Vg] = svd(Jg,'econ');
    G = Ug * Vg';

    % Update F
    F = zeros(numclass,numsample);
    for i = 1:numsample
        Dis = zeros(numclass,1);
        for j = 1:numclass
            Dis(j) = (norm(Z(:,i) - G(:,j)))^2;
        end
        [~, r] = min(Dis);
        F(r(1), i) = 1;
    end
    f_F = zeros(numclass,numsample);
    for ii = 1:numsample
        diff = f_G - repmat(Z_fp(:,ii),1,numclass);
        distsf = sum(diff.^2,1);
        [~, idxm] = min(distsf);
        if isempty(idxm), idxm = randi(numclass); end
        f_F(idxm, ii) = 1;
    end

    % Update alpha 
    M = zeros(numview,1);
    for iv = 1:numview
        M(iv) = norm(X{iv} - W{iv} * A * Z, 'fro')^2;
    end
    M(M<=0) = 1e-12;
    invM = 1 ./ M;
    alpha = (invM / sum(invM))';

    % Update gamma 
    eps_safe = 1e-12;
    Fvec = zeros(numview,1);
    for p = 1:numview
        Wrows = sqrt(sum(W{p}.^2,2));
        Fvec(p) = norm(D{p}, 'fro') - sum(Wrows);
        if ~isfinite(Fvec(p)) || Fvec(p) <= 0
            Fvec(p) = eps_safe;
        end
    end
    invF = 1 ./ Fvec;
    invF(~isfinite(invF)) = eps_safe;
    gamma = (invF / sum(invF))';

    % objective
    term_data = 0;
    term_Dnorm = 0;
    term_W21 = 0;
    for p = 1:numview
        resid = X{p} - W{p} * A * (Z + D{p});
        term_data = term_data + alpha(p)^2 * (norm(resid, 'fro')^2);
        term_Dnorm = term_Dnorm + (gamma(p)^2) * norm(D{p}, 'fro');
        W_rows = sqrt(sum(W{p}.^2, 2));
        term_W21 = term_W21 + (1 - gamma(p)^2) * sum(W_rows);
    end
    term_regZ = lambda * norm(Z - G * F, 'fro')^2;
    term_aux = eps_aux * (norm(gamma(:),2)^2);
    obj(iter) = term_data + term_Dnorm + term_W21 + term_regZ + term_aux;

    if (iter == maxIter) || (iter == 15)
        flag = 0;
    end
end

obj = obj(1:iter);

end

