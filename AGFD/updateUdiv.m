function [x_2, error] = updateUdiv(V_v, X_v, U_con, lambda)
% Update U_div for a single view using FISTA with explicit matrix operations (accelerated version).
% The non-accelerated version (using Kronecker product) is commented out below for reference.

[n, k_dim] = size(U_con);  % k_dim = NMF_k

% Vectorize input matrices
x = X_v(:);                % vec(X_v)
u_con = U_con(:);          % vec(U_con)

[d_dim, ~] = size(V_v);   

% Compute residual: y = vec(X_v - U_con * V_v')
U_matrix = reshape(u_con, [n, k_dim]);
Result_matrix = U_matrix * V_v';
y = x - Result_matrix(:);

%% Accelerated version (avoids explicit Kronecker product)
N = n * k_dim;             
error = [];
itermax = 10;
epsilon = 1e-4;
x_0 = zeros(N, 1);
x_1 = zeros(N, 1);
t_0 = 1;

for i = 1:itermax
    % FISTA momentum step
    t_1 = (1 + sqrt(1 + 4 * t_0^2)) / 2;
    z_2 = x_1 + ((t_0 - 1) / t_1) * (x_1 - x_0);

    % Gradient computation without forming large matrix A
    Z_mat = reshape(z_2, [n, k_dim]);        
    Az_mat = Z_mat * V_v';                  
    res_vec = y - Az_mat(:);                

    % Compute A' * residual = Res_mat * V_v
    Res_mat = reshape(res_vec, [n, d_dim]);  
    Grad_mat = Res_mat * V_v;               
    gradient_vec = Grad_mat(:);              

    z_2 = z_2 + gradient_vec;

    alpha = 0.05;  % Fixed step (could be tuned)
    x_2 = sign(z_2) .* max(abs(z_2) - alpha * lambda, 0);

    norm_x2 = norm(x_2);
    if norm_x2 == 0
        norm_x2 = eps;
    end
    error(i, 1) = norm(x_2 - x_1) / norm_x2;       
    X2_mat = reshape(x_2, [n, k_dim]);
    AX2_vec = (X2_mat * V_v')(:);
    error(i, 2) = norm(y - AX2_vec);               

    if error(i, 1) < epsilon || error(i, 2) < epsilon
        break;
    else
        x_0 = x_1;
        x_1 = x_2;
        t_0 = t_1;
    end
end

end

%% Non-accelerated version (uses Kronecker product - slow for large data)

% tic;
% disp('Time for Kronecker-based setup:');
% I_m = speye(n);
% A = kron(V_v, I_m);  % A = V_v ⊗ I_n, size: (n*d) x (n*k)
% y = x - A * u_con;
% toc;
% 
% itermax = 10;
% epsilon = 1e-4;
% N = size(A, 2);  % Should equal n * k_dim
% error = [];
% 
% x_0 = zeros(N, 1);
% x_1 = zeros(N, 1);
% t_0 = 1;
% 
% for i = 1:itermax
%     t_1 = (1 + sqrt(1 + 4 * t_0^2)) / 2;
%     z_2 = x_1 + ((t_0 - 1) / t_1) * (x_1 - x_0);
%     
%     % Explicit gradient: A' * (y - A * z_2)
%     z_2 = z_2 + A' * (y - A * z_2);
%     
%     % Soft thresholding
%     alpha = 1;  % Note: different from accelerated version
%     x_2 = sign(z_2) .* max(abs(z_2) - alpha * lambda, 0);
%     
%     % Error tracking
%     error(i, 1) = norm(x_2 - x_1) / (norm(x_2) + eps);
%     error(i, 2) = norm(y - A * x_2);
%     
%     if error(i, 1) < epsilon || error(i, 2) < epsilon
%         break;
%     else
%         x_0 = x_1;
%         x_1 = x_2;
%         t_0 = t_1;
%     end
% end
