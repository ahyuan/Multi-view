function [P, Y_hat, Y, S, alpha, obj] = RFNIL(X, S_v, beta, gamma, lambda, options)
% P_v dxc; Y_hat nxc; Y_v nxc; S nxn; alpha vx1; 
% X d*N
% initilize
v=options.v;
n=options.n;
c=options.c;
theta = 1e3;        %高斯核  1e2 1e3可用
mu = 1;             %正交约束前系数
epilision = 1e-4;

alpha = ones(v, 1)*(1/v);
S = rand(n, n);
Y_hat = orth(rand(n,c));
Y = cell(v,1);
P = cell(v,1);      %有时初始化为eye?
for i_v = 1:v
    Y{i_v} = rand(n,c);
end

for i_v = 1:v
    d_v = size(X{i_v}, 1);
    P{i_v} = rand(d_v, c);
end

% converage condition
% 初始化辅助变量乘子
G_Yhat = zeros(n,c);
G_Y = cell(v,1);
for i_v = 1:v
    G_Y{i_v} = zeros(n,c);
end
% 初始化 辅助变量
F = Y_hat;
J = Y;

MaxIter=100;
    
    for iter = 1:MaxIter
        iter;
        
        % update S
        S_A = zeros(n,n);S_b = zeros(n,n);S_c = zeros(n,n);
        for i=1:n
            for j=1:n
                 S_A(i,j) = gamma * 0.5 * sum((Y_hat(i,:) - Y_hat(j,:).^2));  
            end
        end

        temp_S_b = zeros(n,n);
        temp_alpha_2sum = 0;
        for i_v = 1:v
            temp_S_b = temp_S_b + alpha(i_v)^2 .* log(S_v{i_v}); 
            temp_alpha_2sum = temp_alpha_2sum+alpha(i_v)^2;
        end
        
        S_b = temp_S_b./temp_alpha_2sum;
        S_c = S_A./temp_alpha_2sum;
        S = exp(S_b+S_c+1);
        S = S(1:n,:)./sum(S(1:n,:),2);
        S;

        % update Y_hat
        % update F
       
        Q = Y_hat - G_Yhat / mu;
        temp_F = rand(n,c);
        for i_v = 1:v
            H_yhat = X{i_v}'*P{i_v} - Y{i_v};
            A = exp(-0.5 * ((H_yhat-F).*(H_yhat-F)) / (theta^2));
            temp_F  = temp_F + alpha(i_v)^2 * (A.*((H_yhat-F)/(theta^2)));
        end
        F = Q + temp_F/mu;
        F;
            % update Y_hat
        L = diag(sum(S,2))-S;
        eta = svd(L);
        L_hat = eta(1)*eye(n,n) - L;
        Y_hat = gpi(L_hat,mu*F+G_Yhat);
%         M_yhat = 2 * gamma * L_hat * Y_hat - mu * F - G_Yhat; %?
%         [svd_U,~,svd_V] = svd(M_yhat,'econ');
%         Y_hat = svd_U * svd_V';
        Y_hat;
            % update G_Yhat
        G_Yhat = G_Yhat + mu*(F - Y_hat);
        G_Yhat;

        % update Y_v
            % update J
       
        for i_v = 1:v
            T_v = X{i_v}'*P{i_v} - Y_hat;
            C = exp(-0.5 * ((J{i_v}-T_v).*(J{i_v}-T_v)) / (theta^2));
            J{i_v} = (Y{i_v}-G_Y{i_v}/mu) - alpha(i_v)^2 .* (C.*((J{i_v}-T_v)/(theta^2))) / mu;
        end
        J;
            % update Y_v
        for i_v = 1:v
            Q_Yv = J{i_v} + G_Y{i_v}/mu;
            Y{i_v} = ComputeSoft(-1 * Q_Yv, beta/mu);
        end
        Y;
            % update G_Y
        for i_v = 1:v
             G_Y{i_v} = G_Y{i_v} + mu * (J{i_v} - Y{i_v});
        end
        G_Y;
    
        % Update P
        for i_v = 1:v
            Q_P = diag(0.5./sqrt(sum(P{i_v}.*P{i_v},2)+epilision)+1e-4); 
            temp_A = X{i_v}'*P{i_v}-(Y{i_v}+Y_hat);
            A_P = exp(-0.5 * (temp_A .* temp_A) / (theta^2));
            P{i_v} = real(alpha(i_v)^2/(1-alpha(i_v)^2) * (Q_P \ (X{i_v} * (A_P .* (-1 * temp_A / (theta^2))) )) );
        end

        P;

        % update alpha
        M_alpha = zeros(v,1);
        for i_v = 1:v
            temp_M = exp(-0.5 * (X{i_v}'* P{i_v} - (Y_hat + Y{i_v})).^2 / (theta^2));
            M_alpha(i_v) = lambda * Compute_KL(S, S_v{i_v}) - sum(temp_M(:)) - Compute_norm21(P{i_v});
        end
        alpha = Compute_SMO(alpha, M_alpha);

      
        alpha;
        % compute obj
        formular1 = 0;
        for i_v = 1:v
            temp1 = 0;
            A = X{i_v}' * P{i_v} - (Y{i_v}+Y_hat);
            A_temp = A.*A;
            [n_A, d_A] = size(A);
            for i = 1:n_A
                for j = 1:d_A
                     temp1 = temp1 + alpha(i_v)^2 * exp(-0.5 * A_temp(i,j) / (theta^2));
                end
            end
           
            temp2 = (1-alpha(i_v)^2) * Compute_norm21(P{i_v});
            temp3 = beta * Compute_norm1(Y{i_v});
            formular1 = formular1 - temp1 + temp2 + temp3;
        end

        formular2 = 0;
        L = diag(sum(S,2))-S;
        formular2 = gamma * trace(Y_hat'*L*Y_hat);

        formular3 = 0;
        for i_v = 1:v
            temp_formular3(i_v) = alpha(i_v)^2 * Compute_KL(S,S_v{i_v});
        end
        
        formular3 = lambda * sum(temp_formular3);
        sumobj = formular1 + formular2 + formular3;

        obj(iter)=sumobj;
        if iter >= 2 && (abs((obj(iter)-obj(iter-1))/obj(iter))<0.0001)
           break;
        end
    
    
    end

end