function score = Compute_norm21(X)

    tmp = sum(X.*X,2);
    tmp = sqrt(tmp);
    score = sum(tmp);

end