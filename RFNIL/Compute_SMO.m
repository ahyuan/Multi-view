function alpha_new = Compute_SMO(alpha, A)

maxiter = 20;
iter = 1;
n = length(alpha);
while(iter < maxiter)

    for i = 1:length(n)

        % 随机选 j
        j = i;
        while (j == i)
            j = randperm(n, 1);
        end
        L = max(0, alpha(i)+alpha(j)-1); H = min(1, alpha(i)+alpha(j));
        if L == H  
            continue;
        end
        alpha2_new = A(i)/(A(i)+A(j)) * (alpha(i)+alpha(j));

        if alpha2_new < L
            alpha2_new = L;
        elseif alpha2_new > H
            alpha2_new = H;
        end
        alpha1_new = A(j)/(A(i)+A(j)) * (alpha(i) + alpha2_new);
        alpha(i) = alpha1_new;
        alpha(j) = alpha2_new;
    end
iter = iter + 1;

end
alpha_new  = alpha;
% 
% 
% L = max(0, alpha1+alpha2-1); H = min(1, alpha1+alpha2);
% alpha2_new = a1/(a1+a2) * (alpha1+alpha2);
% if alpha2_new < L
%     alpha2_new= L;
% elseif alpha2_new > H
%     alpha2_new = H;
% end
%    
% alpha1_new = a2/(a1+a2) * (alpha1 + alpha2_new);
% alpha2 = alpha2_new;
% alpha1 = alpha1_new;
% 
% r1 = alpha1;
% r2 = alpha2;

end