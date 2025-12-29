clear;
clc;
warning off;
addpath(genpath('./')); % 加载当前目录下所有子文件夹

% 数据集存储路径
dsPath = 'D:\matlab文件\EE-RCAL\Dataset';
% 结果存储路径
resPath = 'D:\matlab文件\EE-RCAL\result';
% 数据集名称
dataName = 'NGs';
% 检查数据集路径是否存在
if exist(dsPath, 'dir') ~= 7
    fprintf('数据集路径不存在: %s\n', dsPath);
    return; 
end

if exist(resPath, 'dir') ~= 7
    fprintf('结果存储路径不存在: %s\n', resPath);
    return; 
end


dataPathFull = fullfile(dsPath, [dataName, '.mat']);
if exist(dataPathFull, 'file') ~= 2
    fprintf('在路径中未找到数据文件: %s\n', dataPathFull);
    return;
end


fprintf('Loading dataset: %s\n', dataPathFull);
load(dataPathFull); % 加载数据 X, Y

k = length(unique(Y));
anchor = k;
d = k;
lambda = 0.01;


fprintf('Running EERCAL (lambda=%.4f, d=%d, anchor=%d)...\n', lambda, d, anchor);
try
    [A,W,Z,D,G,F,iter,obj,alpha,gamma] = EERCAL(X, Y, lambda, d, anchor);
catch ME
    fprintf('【运行出错】算法执行失败: %s\n', ME.message);
    return;
end


[~, idx] = max(F); % 获取簇标号
idx = idx(:);


if exist('Clustering8Measure_all', 'file') == 2
    res = Clustering8Measure_all(Y, idx);
elseif exist('Clustering8Measure', 'file') == 2
    res = Clustering8Measure(Y, idx);
else
    fprintf('缺失 Clustering8Measure 函数，无法计算指标。\n');
    return;
end

% 定义8个指标名称
metricNames = {};

fprintf('\n---------------- Results for %s ----------------\n', dataName);
for i = 1:min(length(metricNames), length(res))
    fprintf('%-10s: %.6f\n', metricNames{i}, res(i));
end
fprintf('--------------------------------------------------\n');

saveName = fullfile(resPath, [dataName, '_result.mat']);
save(saveName, 'res', 'idx', 'F', 'iter', 'obj', 'lambda', 'anchor');
fprintf('Results saved to: %s\n', saveName);
fprintf('Done.\n');