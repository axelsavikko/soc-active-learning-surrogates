function s = normalizeSpatialTargets(Y)
    s.mu = mean(Y, "all");
    s.sigma = std(Y, 0, "all");
    s.sigma(s.sigma == 0) = 1;
    Ynorm = (Y - s.mu) ./ s.sigma;
end