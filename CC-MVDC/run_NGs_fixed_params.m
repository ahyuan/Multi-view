clear
clc

%% Fixed parameters
alpha = 1e-3;
beta  = 1;
gamma = 1e-3;
sigma = 3;
maxIter = 50;
retry = 10;

%% Load NGs dataset
current_dir = fileparts(mfilename('fullpath'));
datasets_dir = fullfile(current_dir, 'datasets');
path = fullfile(datasets_dir, 'NGs.mat');

data = load(path);
X = data.X;
Y_real = data.Y;
clu_num = max(Y_real);
view_num = length(X);

for v = 1:view_num
    X{v} = X{v}';
    X{v} = NormalizeFea(X{v}, 0);
end

[~, n] = size(X{1});

%% Run experiments
ACC_list = zeros(retry, 1);
NMI_list = zeros(retry, 1);
F_list   = zeros(retry, 1);
AR_list  = zeros(retry, 1);
RI_list  = zeros(retry, 1);

for t = 1:retry
    [~, Y_p, Y, mu, R, S, obj] = MDCAW(X, alpha, beta, gamma, sigma, clu_num, maxIter);

    Y_pre = zeros(n, 1);
    for i = 1:n
        [~, k] = max(Y(i, :));
        Y_pre(i) = k;
    end

    acc = Accuracy(Y_pre, Y_real);
    [~, nmi, ~] = compute_nmi(Y_real, Y_pre);
    [f, ~, ~] = compute_f(Y_real, Y_pre);
    [AR, RI, ~, ~] = RandIndex(Y_real, Y_pre);

    ACC_list(t) = acc;
    NMI_list(t) = nmi;
    F_list(t)   = f;
    AR_list(t)  = AR;
    RI_list(t)  = RI;
end

%% Output results
fprintf('========== NGs dataset (alpha=%.2f, beta=%.2f, gamma=%.2f) ==========\n', alpha, beta, gamma);
fprintf('ACC:  mean = %.4f, std = %.4f\n', mean(ACC_list), std(ACC_list));
fprintf('NMI:  mean = %.4f, std = %.4f\n', mean(NMI_list), std(NMI_list));
fprintf('Fscore:    mean = %.4f, std = %.4f\n', mean(F_list),   std(F_list));
fprintf('ARI:   mean = %.4f, std = %.4f\n', mean(AR_list),  std(AR_list));
%fprintf('RI:   mean = %.4f, std = %.4f\n', mean(RI_list),  std(RI_list));
fprintf('====================================================================\n');
