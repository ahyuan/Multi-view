function score = Compute_KL(S, S_v)

score_KL = sum(sum(S.*log(eps + S./(S_v+eps))));

score = score_KL;

end