function [objectives, score, index] = AGFD(X, beta, r, tau, NITER, NMF_k, vN)
% Inputs:
%   X        - Cell array of input data matrices {X1, X2, ..., XvN}
%   beta     - Regularization parameter for graph  term
%   r        - Exponent parameter for adaptive weight learning
%   tau      - Regularization parameter for U_con
%   NITER    - Maximum number of iterations
%   NMF_k    - Latent dimensionality after matrix factorization
%   vN       - Number of views
%
% Outputs:
%   objectives - Objective function values over iterations
%   score      - Feature importance scores
%   index      - Sorted indices of features by descending score

% Initialize consistency weights uniformly
Alpha = ones(vN, 1) / vN;

[n, ~] = size(X{1});

% Preallocate cell arrays
U_div = cell(1, vN);          % View-specific basis matrices
U_con = abs(rand(n, NMF_k)) * 0.1;  % Shared consensus basis

V = cell(1, vN);              % Coefficient matrices per view
SS = cell(1, vN);             % Similarity matrices per view
LL = cell(1, vN);             % Graph Laplacians per view

% Concatenate all views horizontally for global similarity construction
NorX = [];
for i = 1:length(X)
    NorX = [NorX, X{i}];
end

% Graph construction options
options = struct();
options.NeighborMode = 'KNN';
options.k = 5;
options.WeightMode = 'HeatKernel';
options.t = 5;

% Build global similarity matrix and Laplacian
S = constructW(NorX, options);
S = NormalizeFea(S, 1);
L = diag(sum(S, 1)) - S;
S = zeros(n, n);  % Will be updated later per iteration

% GPU detection
use_gpu = (gpuDeviceCount > 0);
if use_gpu
    fprintf('Using GPU to initialize U_div and V...\n');
else
    fprintf('No GPU available; using CPU...\n');
end

% Initialize per-view variables
for vIndex = 1:vN
    [n, d] = size(X{vIndex});
    
    if use_gpu
        U_div{vIndex} = gpuArray.rand(n, NMF_k);
        V{vIndex}      = abs(gpuArray.rand(d, NMF_k));
    else
        U_div{vIndex} = rand(n, NMF_k);
        V{vIndex}     = abs(rand(d, NMF_k));
    end
    
    % Build view-specific similarity and Laplacian
    SS_cpu = constructW(X{vIndex}, options);
    SS_cpu = NormalizeFea(SS_cpu, 1);
    SS{vIndex} = SS_cpu;
    LL{vIndex} = diag(sum(SS_cpu, 1)) - SS_cpu;
end

% Main optimization loop
diff = 1;
iteration = 1;
objectives = zeros(NITER, 1);
D_kl = zeros(vN, 1);

T = cell(vN, 1);
for i = 1:vN
    T{i} = zeros(n, n);
end

T1_iter = cell(1, NITER);
T2_iter = cell(1, NITER);

while (iteration <= NITER && diff > 0.1)
    tic;
    
    %% Step 1: Update Alpha 
    temp = zeros(vN, 1);
    U_con = gather(U_con); 
    
    for vIndex = 1:vN
        U_div{vIndex} = gather(U_div{vIndex});
        V{vIndex}     = gather(V{vIndex});
        
        reconstruction_error = norm(X{vIndex} - (U_con + U_div{vIndex}) * V{vIndex}', 'fro')^2;
        sparsity_V           = sum(sqrt(sum(V{vIndex} .^ 2, 2)));
        sparsity_Udiv        = norm(U_div{vIndex}, 1);
        
        temp(vIndex) = reconstruction_error - sparsity_V + sparsity_Udiv + D_kl(vIndex);
        temp(vIndex) = abs(temp(vIndex));
    end

    exponent = 1 / (1 - r);
    Alpha = (temp .^ exponent) / sum(temp .^ exponent + eps);
    
    %% Step 2: Update V 
    for vIndex = 1:vN
        row_norms = sqrt(sum(V{vIndex} .^ 2, 2) + eps);
        D_weight  = diag(0.5 ./ row_norms);
        
        numerator   = 2 * (Alpha(vIndex)^r) * X{vIndex}' * (U_con + U_div{vIndex});
        denominator = 2 * (Alpha(vIndex)^r) * V{vIndex} * (U_con' + U_div{vIndex}') * (U_con + U_div{vIndex}) ...
                      + (1 - Alpha(vIndex)^r) * D_weight * V{vIndex};
        
        V{vIndex} = V{vIndex} .* (numerator ./ (denominator + eps));
        V{vIndex} = max(V{vIndex}, 0); 
    end
    
    %% Step 3: Update U_div
    for vIndex = 1:vN
        [x, ~] = updateUdiv(V{vIndex}, X{vIndex}, U_con, Alpha(vIndex)^r);
        U_div{vIndex} = reshape(x, [n, NMF_k]);
    end
    
    %% Step 4: Update U_con
    T1 = zeros(n, NMF_k);
    T2 = zeros(n, NMF_k);
    for vIndex = 1:vN
        T1 = T1 + (Alpha(vIndex)^r) * X{vIndex} * V{vIndex};
        T2 = T2 + (Alpha(vIndex)^r) * ((U_con + U_div{vIndex}) * (V{vIndex}' * V{vIndex}) + LL{vIndex} * U_con);
    end
    T1_iter{iteration} = T1;
    T2_iter{iteration} = T2;
    
    numerator_Ucon   = T1 + 2 * tau * U_con;
    denominator_Ucon = T2 + beta * L * U_con + 2 * tau * U_con + eps;
    U_con = U_con .* (numerator_Ucon ./ denominator_Ucon);
    U_con = max(U_con, 0);
    
    %% Step 5: Update global similarity matrix S
    disp('Updating S...');
    tic;
    
    % Original slow implementation
    % for i = 1:n
    %     for j = 1:n
    %         aij = 0.5 * beta * norm(U_con(i, :) - U_con(j, :), 2);
    %         t_up = 0;
    %         t_down = 0;
    %         for vIndex = 1:vN
    %             t_up = t_up + Alpha(vIndex)^r * (log(SS{vIndex}(i, j) + eps) - 1);
    %             t_down = t_down + Alpha(vIndex)^r;
    %         end
    %         S(i, j) = exp((-aij + t_up) / t_down);
    %     end
    %     S(i, :) = S(i, :) / sum(S(i, :));
    % end
    
    % Use GPU-accelerated version
    S = update_S_gpu(U_con, SS, Alpha, r, beta, eps);
    S = gather(S);
    S = (S + S') / 2;               
    S = NormalizeFea(S, 1);         
    L = diag(sum(S, 1)) - S;        
    
    for vIndex = 1:vN
        D_kl(vIndex) = sum(sum(S .* log((S + eps) ./ (SS{vIndex} + eps))));
    end
    toc;
    
    %% Step 6: Evaluate objective function
    sum_term1 = 0;  
    sum_term2 = 0;  
    sum_term3 = 0;  
    sum_term4 = 0; 
    sum_term5 = 0;  
    sum_term6 = 0; 
    for vIndex = 1:vN
        sum_term1 = sum_term1 + Alpha(vIndex)^r * norm(X{vIndex} - (U_con + U_div{vIndex}) * V{vIndex}', 'fro')^2;
        sum_term2 = sum_term2 + (1 - Alpha(vIndex)^r) * sum(sqrt(sum(V{vIndex} .^ 2, 2)));
        sum_term3 = sum_term3 + Alpha(vIndex)^r * D_kl(vIndex);
        %sum_term4 = sum_term4 + Alpha(vIndex)^r * trace(U_con' * L * U_con);
        sum_term6 = sum_term6 + Alpha(vIndex)^r * norm(U_div{vIndex}, 1);
    end
    sum_term5 = beta * trace(U_con' * L * U_con);  
    
    Tempobj = sum_term1 + sum_term2 + sum_term3 + sum_term5 + sum_term6;
    objectives(iteration) = Tempobj;
    
    if iteration > 1
        diff = abs(objectives(iteration - 1) - objectives(iteration));
    end
    
    elapsed_iteration = toc;
    fprintf('Iteration %d | Objective: %.6f | Time: %.2f sec\n', ...
            iteration, objectives(iteration), elapsed_iteration);
    iteration = iteration + 1;
end

% Aggregate all V matrices for feature scoring
Vcon = [];
for vIndex = 1:vN
    Vcon = [Vcon; V{vIndex}];
end

score = sum(Vcon .* Vcon, 2);  % L2-norm squared per feature
[~, index] = sort(score, 'descend');  % Rank features by importance

end
