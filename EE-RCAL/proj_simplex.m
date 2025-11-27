function x = proj_simplex(v)
% 将向量 v 投影到概率 simplex { x >= 0, sum(x) = 1 }
v = v(:);
m = length(v);
if all(abs(v - v(1)) < 1e-12)
    x = ones(m,1) ./ m; return;
end
[vs, ~] = sort(v, 'descend');
css = cumsum(vs);
rho = find(vs - (css - 1) ./ (1:m)' > 0, 1, 'last');
if isempty(rho)
    theta = (css(end) - 1) / m;
else
    theta = (css(rho) - 1) / rho;
end
x = max(v - theta, 0);
s = sum(x);
if s <= 0 || ~isfinite(s)
    x = ones(m,1) ./ m;
else
    x = x / s;
end
x(~isfinite(x) | x < 0) = 0;
s = sum(x);
if s <= 0
    x = ones(m,1) ./ m;
else
    x = x / s;
end
end
