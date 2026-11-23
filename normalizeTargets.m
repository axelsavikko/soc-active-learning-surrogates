function [s, Ynorm] = normalizeTargets(Y)
    s.mu = mean(Y, 1);
    s.sigma = std(Y, 0, 1);
    s.sigma(s.sigma == 0) = 1;
    Ynorm = (Y - s.mu) ./ s.sigma;
end