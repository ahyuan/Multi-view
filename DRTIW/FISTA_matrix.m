function [X_opt, f_values] = FISTA_matrix(A, B, lambda, max_iter, tol)
    % 输入:
    % A: 矩阵 A (m x n)
    % B: 矩阵 B (m x p)
    % lambda: 正则化参数
    % max_iter: 最大迭代次数
    % tol: 收敛阈值

    % 初始化
    [m, n] = size(A);
    [~, p] = size(B);
    X0 = zeros(n, p); % 初始点 (n x p)
    Y = X0;
    t = 1;
    L = norm(A' * A, 2); % Lipschitz 常数

    % 存储目标函数值
    f_values = zeros(max_iter, 1);

    % FISTA 迭代
    for k = 1:max_iter
        % 计算梯度 (矩阵形式)
        grad_F = A' * (A * Y - B);

        % 近端算子 (软阈值，矩阵形式)
        X_new = soft_threshold_matrix(Y - (1/L) * grad_F, lambda / L);

        % 更新 t
        t_new = (1 + sqrt(1 + 4 * t^2)) / 2;

        % 更新 Y (矩阵形式)
        Y = X_new + ((t - 1) / t_new) * (X_new - X0);

        % 更新变量
        X0 = X_new;
        t = t_new;

        % 计算目标函数值 (矩阵形式)
        residual = A * X_new - B;
        f_values(k) = 0.5 * norm(residual, 'fro')^2 + lambda * sum(abs(X_new(:)));

        % 检查收敛条件
        if k > 1 && abs(f_values(k) - f_values(k-1))/abs(f_values(k-1)) < tol
            break;
        end
    end

    % 返回最优解和目标函数值
    X_opt = X_new;
    f_values = f_values(1:k);
end

function X = soft_threshold_matrix(Z, lambda)
    % 软阈值函数 (矩阵形式)
    X = sign(Z) .* max(abs(Z) - lambda, 0);
end