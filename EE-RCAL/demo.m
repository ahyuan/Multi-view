% 运行脚本
clear;
clc;
warning off;
addpath(genpath('./'));

% 数据集与路径
dataName = 'NGs';
dsPath = './Dataset/';
resPath = './res-lmd0/';

% 加载数据
fprintf('Loading dataset: %s\n', dataName);
load(fullfile(dsPath, dataName)); % 期望文件内包含 X, Y

k = length(unique(Y));
anchor = k;
d = k;
lambda = 0.01;

fprintf('Running EERCAL (lambda=%.4f, d=%d, anchor=%d)...\n', lambda, d, anchor);

[A,W,Z,D,G,F,iter,obj,alpha,gamma] = EERCAL(X, Y, lambda, d, anchor);

[~, idx] = max(F); % idx 为每个样本的簇标号
idx = idx(:);

if exist('Clustering8Measure_all', 'file') == 2
    res = Clustering8Measure_all(Y, idx);
elseif exist('Clustering8Measure', 'file') == 2
    res = Clustering8Measure(Y, idx);
else
    error('运行错误，缺失函数。');
end

metricNames = {'ACC','NMI','Purity','Fscore'};
fprintf('Results for dataset %s (showing only first 4 metrics):\n', dataName);
for i = 1:min(length(metricNames), length(res))
    fprintf('%s: %.6f\n', metricNames{i}, res(i));
end

fprintf('Done.\n');
