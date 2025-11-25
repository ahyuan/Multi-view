function score = Compute_norm1(X)

X =  abs(X);
score = sum(X(:));
end