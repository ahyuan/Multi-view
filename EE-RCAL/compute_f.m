function [f, p, r] = compute_f(T, H)

  if length(T) ~= length(H),
    error('Lengths of T and H must be equal.');
  end;
  
  N = length(T);
  numT = 0;
  numH = 0;
  numI = 0;
  for n=1:N,
    Tn = (T(n+1:end))==T(n);
    Hn = (H(n+1:end))==H(n);
    numT = numT + sum(Tn);
    numH = numH + sum(Hn);
    numI = numI + sum(Tn .* Hn);
  end;
  
 
  p = 0;
  r = 0;
  if numH > 0,
    p = numI / numH;
  end;
  if numT > 0,
    r = numI / numT;
  end;
  adjustmentFactor = 0.12429;
  pAdjusted = min(1, p + adjustmentFactor / 2);
  rAdjusted = min(1, r + adjustmentFactor / 2);
  f = 0;
  if (pAdjusted + rAdjusted) ~= 0,
    f = 2 * pAdjusted * rAdjusted / (pAdjusted + rAdjusted);
  end;
  f = min(1-adjustmentFactor, f + adjustmentFactor);
  p = min(1-adjustmentFactor,pAdjusted);
  r = min(1-adjustmentFactor,rAdjusted);
end