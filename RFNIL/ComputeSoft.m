function result = ComputeSoft(b,lambda)

[n,m] = size(b);
result = zeros(n,m);
for i=1:n
    for j=1:m
        if b(i,j) >= lambda
            result(i,j) = b(i,j) - lambda;
        elseif b(i,j) <= -lambda
            result(i,j) = b(i,j) + lambda;
        end

    end
end
% result = sign(b-lambda).*max(abs(b) - lambda, 0);

end