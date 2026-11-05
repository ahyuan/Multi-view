function w_opt = optimize_w_analytic(a, b, mu, beta)

    V = length(a);
    A = a(:);               
    B = (beta / mu) * b(:); 
    eps_b = 1e-12;
    B = max(B, eps_b);
    if all(abs(A - mean(A)) < 1e-12)
        w_opt = ones(V,1) / V;
        return;
    end

    min_A = min(A);
    lambda_high = min_A - 1e-10;
    lambda_low = -1e8;
    f = @(lam) sum( sqrt( B ./ (A - lam) ) );
    while f(lambda_low) < 1
        lambda_low = lambda_low * 2;
        if lambda_low < -1e16
            break;
        end
    end
    while f(lambda_high) > 1 && (min_A - lambda_high) > 1e-12
        lambda_high = (lambda_high + min_A) / 2;
    end
    tol = 1e-12;
    max_iter = 1000;
    for iter = 1:max_iter
        lam_mid = (lambda_low + lambda_high) / 2;
        f_mid = f(lam_mid);
        if abs(f_mid - 1) < tol
            break;
        elseif f_mid > 1
            lambda_low = lam_mid;
        else
            lambda_high = lam_mid;
        end
    end
    lam_opt = (lambda_low + lambda_high) / 2;
    w_opt = sqrt( B ./ (A - lam_opt) );
    w_opt = w_opt / sum(w_opt);
    w_opt = max(w_opt, 1e-10);
    w_opt = w_opt / sum(w_opt);
end