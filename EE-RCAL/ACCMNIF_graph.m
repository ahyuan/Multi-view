clear;
clc;
warning off;
addpath(genpath('./'));

%% 1. 全局设置
% -------------------------------------------------------------------------
% 数据集列表
% -------------------------------------------------------------------------
dsList = {'ORL_mtv', 'NGs', 'Hdigit', 'WebKB_2views', 'NUSWIDEOBJ', ...
          'Caltech101-7', 'Caltech101-20', 'Caltech101-all_fea'}; 
      
% 路径设置
dsPath = './Dataset/';
saveDir = 'D:\matlab文件\EE-RCAL\res';       % 原有的最佳结果保存路径
graphDir = 'D:\matlab文件\EE-RCAL\res_graph'; % [新增] 用于画图的所有数据保存路径

% 自动创建文件夹
if ~exist(saveDir, 'dir'), mkdir(saveDir); end
if ~exist(graphDir, 'dir'), mkdir(graphDir); end

%% 2. 自动循环所有数据集
for dsi = 1:length(dsList)
    dataName = dsList{dsi};
    
    % --- 清理工作区 ---
    clear X Y Z F_mat Best_E2E Best_Spec
    
    fprintf('\n================================================================================\n');
    fprintf('PROCESSING DATASET (%d/%d): %s\n', dsi, length(dsList), dataName);
    fprintf('================================================================================\n');
    try
        % --- 加载数据 ---
        fprintf('Loading dataset: %s\n', dataName);
        if exist(fullfile(dsPath, [dataName, '.mat']), 'file')
            load(fullfile(dsPath, dataName)); 
        elseif exist(fullfile(dsPath, dataName), 'file')
            load(fullfile(dsPath, dataName));
        else
            warning('未找到数据集文件 %s，跳过该数据集。', dataName);
            continue; 
        end
        
        % --- 预处理标签 ---
        Y = double(Y);
        if size(Y, 1) == 1, Y = Y'; end
        if min(Y) < 1, Y = Y - min(Y) + 1; end
        k = length(unique(Y));
        
        % --- 参数设置 ---
        anchor_ratios = [1, 2, 3];     % 对应 k, 2k, 3k
        lambdas = [0.001, 0.01, 0.1, 1, 10, 100, 1000];
        d = k;
        
        % [新增] 初始化全量记录矩阵 (行: Anchor, 列: Lambda)
        num_anc = length(anchor_ratios);
        num_lam = length(lambdas);
        
        % E2E 方法的全量记录
        All_ACC_E2E    = zeros(num_anc, num_lam);
        All_NMI_E2E    = zeros(num_anc, num_lam);
        All_Fscore_E2E = zeros(num_anc, num_lam);
        
        % Spectral 方法的全量记录
        All_ACC_Spec    = zeros(num_anc, num_lam);
        All_NMI_Spec    = zeros(num_anc, num_lam);
        All_Fscore_Spec = zeros(num_anc, num_lam);
        
        % --- 初始化最佳结果记录容器 (保留原逻辑) ---
        Best_E2E.ACC=[-1,0,0]; Best_E2E.NMI=[-1,0,0]; Best_E2E.Purity=[-1,0,0]; Best_E2E.Fscore=[-1,0,0];
        Best_Spec.ACC=[-1,0,0]; Best_Spec.NMI=[-1,0,0]; Best_Spec.Purity=[-1,0,0]; Best_Spec.Fscore=[-1,0,0];
        
        total_time = 0;
        run_count = 0;
        
        fprintf('Starting Grid Search for %s...\n', dataName);
        
        %% 3. 网格搜索循环
        for ar = 1:length(anchor_ratios)
            anchor = anchor_ratios(ar) * k;
            
            for lm = 1:length(lambdas)
                lambda = lambdas(lm);
                
                t_start = tic;
                
                try
                    % ---------------------- 运行核心算法 ----------------------
                    [~, ~, Z, ~, ~, F_mat, ~, ~, ~, ~] = EERCAL(X, Y, lambda, d, anchor);
                    
                    t_cost = toc(t_start);
                    total_time = total_time + t_cost;
                    run_count = run_count + 1;
                    
                    % =========================================================
                    % 方法 A: End-to-End (E2E)
                    % =========================================================
                    % 注意：聚类标签通常按行取最大值
                    [~, idx_e2e] = max(F_mat, [], 2); 
                    
                    if exist('Clustering8Measure_all', 'file')
                        res_e2e = Clustering8Measure_all(Y, idx_e2e);
                    elseif exist('Clustering8Measure', 'file')
                        res_e2e = Clustering8Measure(Y, idx_e2e);
                    else
                        res_e2e = zeros(1,4); 
                    end
                    
                    % 提取指标
                    curr_E2E_ACC = res_e2e(1); 
                    curr_E2E_NMI = res_e2e(2); 
                    curr_E2E_Pur = res_e2e(3);
                    if length(res_e2e) >= 4, curr_E2E_Fsc = res_e2e(4); else, curr_E2E_Fsc = 0; end
                    
                    % [新增] 存入全量矩阵
                    All_ACC_E2E(ar, lm)    = curr_E2E_ACC;
                    All_NMI_E2E(ar, lm)    = curr_E2E_NMI;
                    All_Fscore_E2E(ar, lm) = curr_E2E_Fsc;
                    
                    % 更新最佳值 (原逻辑)
                    if curr_E2E_ACC > Best_E2E.ACC(1), Best_E2E.ACC = [curr_E2E_ACC, anchor, lambda]; end
                    if curr_E2E_NMI > Best_E2E.NMI(1), Best_E2E.NMI = [curr_E2E_NMI, anchor, lambda]; end
                    if curr_E2E_Pur > Best_E2E.Purity(1), Best_E2E.Purity = [curr_E2E_Pur, anchor, lambda]; end
                    if curr_E2E_Fsc > Best_E2E.Fscore(1), Best_E2E.Fscore = [curr_E2E_Fsc, anchor, lambda]; end
                    
                    % =========================================================
                    % 方法 B: Spectral Clustering (Spec)
                    % =========================================================
                    [U_spec, ~, ~] = svd(Z', 'econ'); 
                    if size(U_spec, 2) >= k, U_spec = U_spec(:, 1:k); end
                    U_norm = U_spec ./ repmat(sqrt(sum(U_spec.^2, 2) + eps), 1, k);
                    
                    max_acc_spec = -1;
                    res_spec_best = zeros(1,4);
                    retry_times = 20; 
                    
                    for r = 1:retry_times
                        try
                            if exist('litekmeans', 'file')
                                idx_spec = litekmeans(U_norm, k, 'MaxIter', 100, 'Replicates', 1);
                            else
                                idx_spec = kmeans(U_norm, k, 'MaxIter', 100, 'Replicates', 1);
                            end
                            
                            if exist('Clustering8Measure_all', 'file')
                                temp_res = Clustering8Measure_all(Y, idx_spec);
                            else
                                temp_res = Clustering8Measure(Y, idx_spec);
                            end
                            
                            if temp_res(1) > max_acc_spec
                                max_acc_spec = temp_res(1);
                                res_spec_best = temp_res;
                            end
                        catch
                            continue;
                        end
                    end
                    
                    curr_Spec_ACC = res_spec_best(1); 
                    curr_Spec_NMI = res_spec_best(2);
                    curr_Spec_Pur = res_spec_best(3); 
                    if length(res_spec_best) >= 4, curr_Spec_Fsc = res_spec_best(4); else, curr_Spec_Fsc = 0; end
                    
                    % [新增] 存入全量矩阵
                    All_ACC_Spec(ar, lm)    = curr_Spec_ACC;
                    All_NMI_Spec(ar, lm)    = curr_Spec_NMI;
                    All_Fscore_Spec(ar, lm) = curr_Spec_Fsc;
                    
                    % 更新最佳值 (原逻辑)
                    if curr_Spec_ACC > Best_Spec.ACC(1), Best_Spec.ACC = [curr_Spec_ACC, anchor, lambda]; end
                    if curr_Spec_NMI > Best_Spec.NMI(1), Best_Spec.NMI = [curr_Spec_NMI, anchor, lambda]; end
                    if curr_Spec_Pur > Best_Spec.Purity(1), Best_Spec.Purity = [curr_Spec_Pur, anchor, lambda]; end
                    if curr_Spec_Fsc > Best_Spec.Fscore(1), Best_Spec.Fscore = [curr_Spec_Fsc, anchor, lambda]; end
                    
                    fprintf('%-6d %-8.4f | %.4f / %.4f    | %.4f / %.4f    | %.4f\n', ...
                        anchor, lambda, curr_E2E_ACC, curr_E2E_NMI, curr_Spec_ACC, curr_Spec_NMI, t_cost);
                    
                catch ME
                    fprintf('Error at anchor=%d, lambda=%.4f: %s\n', anchor, lambda, ME.message);
                end
            end
        end
        
        %% 4. 保存结果
        if run_count > 0, avg_time = total_time / run_count; else, avg_time = 0; end
        
        % (1) 保存最佳结果 (保留你原有的保存逻辑)
        saveBestName = fullfile(saveDir, [dataName, '_BestResult.mat']);
        save(saveBestName, 'Best_E2E', 'Best_Spec', 'anchor_ratios', 'lambdas', 'avg_time');
        
        % (2) [新增] 保存全量结果 (用于画图) 到 res_graph 文件夹
        saveAllName = fullfile(graphDir, [dataName, '_AllResults.mat']);
        save(saveAllName, ...
             'All_ACC_E2E', 'All_NMI_E2E', 'All_Fscore_E2E', ...
             'All_ACC_Spec', 'All_NMI_Spec', 'All_Fscore_Spec', ...
             'anchor_ratios', 'lambdas');
         
        fprintf('Best Results saved to: %s\n', saveBestName);
        fprintf('Full Results (for Plotting) saved to: %s\n', saveAllName);
        
        %% 5. 打印对比
        fprintf('\n[FINAL BEST %s]\n', dataName);
        fprintf('E2E_ACC: %.4f, Spec_ACC: %.4f\n', Best_E2E.ACC(1), Best_Spec.ACC(1));
        
    catch ME_Dataset
        fprintf('\n!!! Error processing dataset %s: %s\n', dataName, ME_Dataset.message);
    end
end
fprintf('\nAll Done.\n');