function score = Compute_H(S, S_v)

score_H = sum(sum(S_v.*log(eps + S)));

score = -score_H;

end