function newIDs = selectHybridSims(score, acquiredIDs, batchSize)
    eligibleIDs = setdiff(1:numel(score), acquiredIDs);
    assert(numel(eligibleIDs) >= batchSize, ...
        'Not enough unacquired simulations.');

    [~, order] = sort(score(eligibleIDs), 'descend');
    rankedIDs = eligibleIDs(order);

    nTop = floor(batchSize / 2);
    nSpread = batchSize - nTop;
    spreadRanks = floor(linspace(nTop + 1, numel(rankedIDs), nSpread));

    newIDs = [rankedIDs(1:nTop), rankedIDs(spreadRanks)];
    newIDs = newIDs(:).';
end