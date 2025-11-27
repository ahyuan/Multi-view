function [A,W,Z,D,G,F,iter,obj,alpha,gamma] = EERCAL(X,Y,lambda,d,numanchor,varargin)
% EERCAL: Anchor-Expressive End-to-end multi-view clustering

% Parse optional inputs
p = inputParser;
addParameter(p,'maxIter',50,@(x)isnumeric(x)&&x>0);
addParameter(p,'tol',1e-6,@isnumeric);
parse(p,varargin{:});
maxIter = p.Results.maxIter;
tol = p.Results.tol;

rng('default');

V = numel(X);
n = size(Y,1);
m = numanchor;
classes = unique(Y);
k = length(classes);

% Normalize each view (zero mean, unit variance per feature)
for v = 1:V
Xv = X{v};
Xv = Xv'; % n x d_p
mu = mean(Xv,1);
sigma = std(Xv,0,1);
sigma(sigma < 1e-12) = 1;
Xv = (Xv - mu) ./ sigma;
X{v} = Xv'; % d_p x n
end

% Initialize variables
W = cell(V,1);
D = cell(V,1);
for v = 1:V
dp = size(X{v},1);
% random orthonormal initialization for W{v} (d_p x d)
R = randn(dp,d);
if dp >= d
    [Q,] = qr(R,0); W{v} = Q(:,1:d);
else
[Q,] = qr([R; randn(d-dp,d)],0); W{v} = Q(1:dp,1:d);
end
D{v} = ones(m,n) ./ m; % each column sums to 1
end

% A
[Ua,~] = qr(randn(d,m),0);
A = Ua(:,1:m);

% Z 
Z = ones(m,n) ./ m;

% G
[Ug,~] = qr(randn(m,k),0);
G = Ug(:,1:k);

% F
F = zeros(k,n);
idx = randi(k,n,1);
for i = 1:n, F(idx(i),i) = 1; end

% alpha and gamma initialization (1 x V)
alpha = ones(1,V) / V;
gamma = ones(1,V) / V;

obj = [];
eps_aux = 1e-8;
Delta = 1e-8; % small for Q diag

% Precompute dims
dp_arr = zeros(V,1);
for v = 1:V, dp_arr(v) = size(X{v},1); end

for iter = 1:maxIter
%% 1) Update W^{(p)} 
for v = 1:V
% M^{(p)} = A * (Z + D{p}) (d x n)
Mv = A * (Z + D{v});
% Q^{(p)} diagonal: 1 / sqrt(sum_j W_{ij}^2 + theta)
Wv = W{v};
sq = sum(Wv.^2,2);
theta = 1e-8;
Qdiag = 1 ./ sqrt(sq + theta);
Qp = diag(Qdiag);
% compute eta: largest eigenvalue of (1 - gamma^2)*Qp
lam = (1 - gamma(v)^2);
if lam <= 0
eta = 0;
else
% power method for largest eigenvalue 
eta = lam * max(Qdiag);
end
Qhat = eta * eye(dp_arr(v)) - lam * Qp;
% J^{(p)} = Qhat * W + 2 * alpha^2 * X^{(p)} * Mv'
Jp = Qhat * Wv + 2 * (alpha(v)^2) * (X{v} * Mv');
% SVD and orthogonalize
[Uj,Sj,Vj] = svd(Jp,'econ');
W{v} = Uj * Vj';
end
%% 2) Update A 
J = zeros(d,m);
for v = 1:V
    Mv = Z + D{v};
    tmp = 2 * (alpha(v)^2) * (W{v}' * X{v} * Mv');
    J = J + tmp;
end
[Ua,Sa,Va] = svd(J,'econ');
if isempty(Ua) || isempty(Va)
    % fallback
    [Ua,~] = qr(randn(d,m),0);
    A = Ua(:,1:m);
else
    A = Ua * Va';
end

%% 3) Update Z 
denom = sum(alpha.^2) + lambda;
for j = 1:n
    numer = zeros(m,1);
    for v = 1:V
        % M^{(p)}_{:,j} = X^{(p)}(:,j) - W^{(p)} * A * D^{(p)}(:,j)
        Mcol = X{v}(:,j) - W{v} * (A * D{v}(:,j));
        numer = numer + alpha(v)^2 * (A' * W{v}' * Mcol);
    end
    numer = numer + lambda * (G * F(:,j));
    pj = numer ./ denom;
    % project pj to simplex
    Z(:,j) = proj_simplex(pj);
end

%% 4) Update D^{(p)} 
for v = 1:V
    denomD = 1 + 4 * (gamma(v)^2);
    WA = W{v} * A; % d_p x m
    % compute base term T = X^{(p)} - 0.5 * W^{(p)} * A * C
    T = X{v} - 0.5 * (WA * Z);
    % For each column, compute qi and project
    % qi = (2/denomD) * (T(:,j)' * WA)'.  (m x 1)
    % vectorized:
    Qall = (2/denomD) * (WA' * T); % m x n
    % project each column to simplex
    for j = 1:n
        qj = Qall(:,j);
        D{v}(:,j) = proj_simplex(qj);
    end
end

%% 5) Update G
Jg = Z * F';
[Ug,~] = qr(Jg,0);
% To ensure G is m x k with orthonormal columns we take first k columns
if size(Ug,2) < k
    % pad
    [Qpad,~] = qr(randn(m,k),0);
    G = Qpad(:,1:k);
else
    G = Ug(:,1:k);
end
% refine via SVD for exact solution (U*V')
[Ug2,~,Vg2] = svd(Jg,'econ');
G = Ug2(:,1:k) * Vg2(:,1:k)';

%% 6) Update F 
F = zeros(k,n);
for j = 1:n
    % choose cluster whose centroid G(:,i) closest to Z(:,j)
    diffs = sum((G - Z(:,j)).^2,1); % 1 x k
    [~, idx_min] = min(diffs);
    F(idx_min,j) = 1;
end

%% 7) Update alpha 
R = zeros(V,1);
for v = 1:V
    R(v) = norm(X{v} - W{v} * (A * (Z + D{v})), 'fro')^2;
end
R(R <= 0) = 1e-12;
invR = 1 ./ R;
if ~all(isfinite(invR))
    invR(~isfinite(invR)) = max( min(invR(isfinite(invR))), 1e-12 );
end
alpha = (invR / sum(invR))'; % make it column
alpha = alpha(:)'; % 1 x V

%% 8) Update gamma 
Fp = zeros(V,1);
for v = 1:V
    Fp(v) = norm(D{v}, 'fro') - sum(sqrt(sum(W{v}.^2,2)));
    if ~isfinite(Fp(v)) || Fp(v) <= 0
        Fp(v) = 1e-12 + abs(Fp(v));
    end
end
invFp = 1 ./ Fp;
if ~all(isfinite(invFp))
    invFp(~isfinite(invFp)) = max( min(invFp(isfinite(invFp))), 1e-12 );
end
gamma = (invFp / sum(invFp))';
gamma = gamma(:)';

%% 9) Compute objective value 
term_data = 0;
term_Dnorm = 0;
term_W21 = 0;
for v = 1:V
    resid = X{v} - W{v} * (A * (Z + D{v}));
    term_data = term_data + alpha(v)^2 * norm(resid,'fro')^2;
    term_Dnorm = term_Dnorm + (gamma(v)^2) * norm(D{v}, 'fro');
    W_rows = sqrt(sum(W{v}.^2,2));
    term_W21 = term_W21 + (1 - gamma(v)^2) * sum(W_rows);
end
term_regZ = lambda * norm(Z - G * F, 'fro')^2;
term_aux = eps_aux * (norm(alpha(:),2)^2 + norm(gamma(:),2)^2);
obj(iter) = term_data + term_Dnorm + term_W21 + term_regZ + term_aux;

% Display iteration info (optional)
fprintf('Iter %d, obj=%.6e\n', iter, obj(iter));

% check convergence
if iter > 1 && abs(obj(iter) - obj(iter-1)) < tol
    break;
end
% Trim obj vector
obj = obj(1:iter);

end

