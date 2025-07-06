function [x_2, error] = updateUdiv1(V_v, X_v, U_con, lambda)

[n, k_v] = size(U_con);
% Vectorize the matrices
x = X_v(:); % vec(X^(v))
u_con = U_con(:); % vec(U_con)
% Compute A = V^(v) ⊗ I_m
I_m = speye(n); 
A = kron(V_v,I_m);
y = x - A * u_con;

if nargin < 5
    itermax = 10 ;
end
if nargin <= 4
    epsilon = 1e-4 ;
end

N = size(A,2);
error = [] ;

x_0 = zeros(N,1);
x_1 = zeros(N,1);
t_0 = 1 ;

    for i = 1:itermax
        t_1 = (1+sqrt(1+4*t_0^2))/2 ;
        alpha =1;
        z_2 = x_1 + ((t_0-1)/(t_1))*(x_1 - x_0) ;
        z_2 = z_2+A'*(y-A*z_2);
        x_2 = sign(z_2).*max(abs(z_2)-alpha*lambda,0) ;
        error(i,1) = norm(x_2 - x_1)/norm(x_2) ;
        error(i,2) = norm(y-A*x_2) ;
        if error(i,1) < epsilon || error(i,2) < epsilon
            break;
        else
            x_0 = x_1 ;
            x_1 = x_2 ;
            t_0 = t_1 ;
        end
    end

end

