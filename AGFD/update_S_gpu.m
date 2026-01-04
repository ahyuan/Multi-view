function S = update_S_gpu(U_con, SS, Alpha, r, beta, eps)

    %% ========== 1. GPU 自动判断 ==========
    use_gpu = (gpuDeviceCount > 0);
    if use_gpu
        toArr = @(x) gpuArray(x);
        fprintf('使用 GPU 加速 S 更新...\n');
    else
        toArr = @(x) x;
        fprintf('未检测到 GPU，使用 CPU（仍可分块）...\n');
    end

    %% ========== 2. 基本参数 ==========
    n = size(U_con,1);
    vN = numel(SS);

    %% ========== 3. 转为 GPU (或 CPU) ==========
    U = toArr(U_con);
    Alpha_r = toArr(Alpha.^r);
    t_down = sum(Alpha_r);     % scalar

    %% ========== 4. 预处理 log_SS ==========
    log_SS = cell(1,vN);
    for v = 1:vN
        log_SS{v} = toArr(log(SS{v} + eps) - 1);
    end

    %% ========= 5. 初始化输出 ==========
    S = zeros(n,n, 'like', U);

    %% ========== 6. 设置分块大小 ==========
    % --- 你可以根据显存调大或调小（1024, 2048, 4096）
    block = 1000;

    %% ========== 7. 分块计算 S(i,:) ==========
    for i1 = 1:block:n
        i2 = min(i1 + block - 1, n);

        % ------- (i1:i2,:) 的距离矩阵 -------
        U_block = U(i1:i2, :);
        U_diff = pdist2(U_block, U, 'euclidean');  % (i_block × n)
        aij = 0.5 * beta * U_diff;

        % ------- 计算 t_up -------
        t_up = zeros(size(aij), 'like', U);
        for v = 1:vN
            t_up = t_up + Alpha_r(v) * log_SS{v}(i1:i2,:);
        end

        % ------- 计算 S -------
        S(i1:i2,:) = exp((-aij + t_up) ./ t_down);

        % ------- 行归一化 -------
        row_sum = sum(S(i1:i2,:), 2);
        S(i1:i2,:) = S(i1:i2,:) ./ row_sum;
    end

    %% ========== 8. 如果用了 GPU，最后 gather 回 CPU ==========
    if use_gpu
        S = gather(S);
    end
end