
seeds = load("Data/seeds.mat").seeds;

if ~isfolder("Results/")
    mkdir("Results/")
end

for currRun=1:5

rng(seeds(currRun));

% Data load
clearvars -except starts increases seeds currRun

labels = load("Data/spatial_candidate_samples.mat").labels;
targets = load("Data/spatial_candidate_samples.mat").targets;

nFeatures = size(labels, 2);

targetStats = struct();

load("Data/training_normalization_spatial.mat")
normalized_labels = (labels - labels_mean) ./ labels_std;

targets = targets(:,2);

simBlockSize = 28*28;
numSims = floor(length(targets) / simBlockSize);

% Reshape into blocks (each column is a block)
label_blocks = reshape(normalized_labels(1:numSims * simBlockSize, :), simBlockSize, numSims, []);
target_blocks = reshape(targets(1:numSims * simBlockSize, :), simBlockSize, numSims, []);

% Reshape unnormalized data into blocks (each column is a block)
label_blocks2 = reshape(labels(1:numSims * simBlockSize, :), simBlockSize, numSims, []);
target_blocks2 = reshape(targets(1:numSims * simBlockSize, :), simBlockSize, numSims, []);

% Shuffle the blocks
shuffledIndices = randperm(numSims);

% Keep track of simulation IDs and select the blocks based on which simID
% is included in the current run
simIDs = 1:numSims;
shuffled_simIDs = simIDs(shuffledIndices);

% First set
start = 40;
oneSet = simBlockSize;

first_set_sims = shuffled_simIDs(1:start);

targetStats.initial = normalizeSpatialTargets(target_blocks(:,first_set_sims,:));

% Comparison

increase = 20;

% Network architecture

mlpLayers = [
    featureInputLayer(15,'Normalization','None')
    fullyConnectedLayer(100)
    reluLayer
    fullyConnectedLayer(100)
    reluLayer
    fullyConnectedLayer(1)
    identityLayer];

options = trainingOptions('adam', ...
    'InitialLearnRate',1e-3, ...
    'MiniBatchSize', 128, ...
    'Verbose', true, ...
    'VerboseFrequency', 250, ...
    'Shuffle','every-epoch', ...
    'ExecutionEnvironment','gpu',...
    'ValidationFrequency',250,...
    "ValidationPatience",10);

mlpOptions = options;
mlpOptions.MaxEpochs = 1000;


% Training loop
tic

numNets = 30;
numCycles = 5;

ensemble = cell(1, numNets);


for i=1:numNets

    fprintf("Base Ensemble, Net %d\n", i)
    sz = numel(first_set_sims);
    shuf = randperm(sz);
    
    shufIDs = first_set_sims(shuf);
    mu = targetStats.initial.mu;
    sigma = targetStats.initial.sigma;

    szTrain = floor(0.9*sz);
    szVal = floor(0.1*sz);
    
    currTrainIDs = shufIDs(1:szTrain);
    currValIDs = shufIDs(szTrain+1:end);
    
    currTrainLabels = reshape(label_blocks(:,currTrainIDs,:), [], nFeatures, 1);
    currTrainTargets = (reshape(target_blocks(:,currTrainIDs,:), [], 1) - mu) ./ sigma;

    currValLabels = reshape(label_blocks(:,currValIDs,:), [], nFeatures, 1);
    currValTargets = (reshape(target_blocks(:,currValIDs,:), [], 1) - mu) ./ sigma;
    
    localOptions = mlpOptions;
    localOptions.ValidationData = {currValLabels, currValTargets};

    trainedNet = trainnet(currTrainLabels, currTrainTargets, mlpLayers,"mae", localOptions);
    ensemble{i} = trainedNet;

end
first_net_time = toc;
% Predicting for all labels with all networks:

results = zeros(simBlockSize, numSims, numNets);

for i=1:numSims
    if mod(i,50) == 0
        fprintf("Simulation %d\n", i)
    end
    currSet = reshape(label_blocks(:,i,:), simBlockSize, nFeatures);
    results(:,i,:) = cell2mat(cellfun(@(net) predict(net,currSet), ensemble, "UniformOutput",false));
end

variance = var(results, 0, 3);
% sum_variance = sum(variance,1)';
% mean_variance = mean(variance, 1)';

%

mean_variance = mean(variance, 1)';
[sortedVars, sortedVarsIdx] = sort(mean_variance, "descend");
plot(1:numSims, sortedVars)

% Choose 20 top variance

alTrainSims = shuffled_simIDs(1:start);
candidatePoolIDs = setdiff(1:numSims, alTrainSims);
[~, alCandidatePoolVarianceIdx] = sort(mean_variance(candidatePoolIDs, :), "descend");
alSelectedIdx = candidatePoolIDs(alCandidatePoolVarianceIdx(1:increase));

alTrainSims = [alTrainSims alSelectedIdx];

targetStats.alTop(1) = normalizeSpatialTargets(target_blocks(:,alTrainSims,:));


al_ensembles = cell(numCycles, numNets);
variances = zeros(simBlockSize, numSims, numCycles);
mean_variances = zeros(numSims,numCycles);
sortedVarsArr = zeros(numSims,numCycles);
sortedVarsIdxArr = zeros(numSims,numCycles);

for k=1:numCycles    
    for i=1:numNets

        fprintf("AL, Cycle %d, Net %d", k, i)
        
        sz = numel(alTrainSims);
        shuf = randperm(sz);
        
        shufIDs = alTrainSims(shuf);
        mu = targetStats.alTop(k).mu;
        sigma = targetStats.alTop(k).sigma;
    
        szTrain = floor(0.9*sz);
        szVal = floor(0.1*sz);
        
        currTrainIDs = shufIDs(1:szTrain);
        currValIDs = shufIDs(szTrain+1:end);
        
        currTrainLabels = reshape(label_blocks(:,currTrainIDs,:), [], nFeatures, 1);
        currTrainTargets = (reshape(target_blocks(:,currTrainIDs,:), [], 1) - mu) ./ sigma;
    
        currValLabels = reshape(label_blocks(:,currValIDs,:), [], nFeatures, 1);
        currValTargets = (reshape(target_blocks(:,currValIDs,:), [], 1) - mu) ./ sigma;
        
        localOptions = mlpOptions;
        localOptions.ValidationData = {currValLabels, currValTargets};

        trainedNet = trainnet(currTrainLabels, currTrainTargets, mlpLayers,"mae", localOptions);
        al_ensembles{k,i} = trainedNet;
    end

    for i=1:numSims
    if mod(i,50) == 0
        fprintf("Simulation %d\n", i)
    end
        currSet = reshape(label_blocks(:,i,:), simBlockSize, nFeatures);
        results(:,i,:) = cell2mat(cellfun(@(net) predict(net,currSet), al_ensembles(k,:), "UniformOutput",false));
    end


    variances(:, :, k) = var(results, 0, 3);
    mean_variances(:,k) = mean(variances(:,:,k), 1);

    [sortedVarsArr(:,k), sortedVarsIdxArr(:,k)] = sort(mean_variances(:,k), "descend");
    
    if k < numCycles
    candidatePoolIDs = setdiff(1:numSims, alTrainSims);
    [~, alCandidatePoolVarianceIdx] = sort(mean_variances(candidatePoolIDs, k), "descend");
    alSelectedIdx = candidatePoolIDs(alCandidatePoolVarianceIdx(1:increase));
    
    alTrainSims = [alTrainSims alSelectedIdx];

    targetStats.alTop(k+1) = normalizeSpatialTargets(target_blocks(:,alTrainSims,:));
    end

end

% % Choose 20 points based on weighted variances
% 
% var_weights = load("var_weights_5.mat").var_weights_5;
% var_weights = reshape(var_weights, 28*28, []);
% 
% weighted_variance = var_weights .* variance;
% sum_weighted_variance = sum(weighted_variance, 1)';
% 
% [sortedWeightedSumVar, sortedWeightedSumVarIdx] = sort(sum_weighted_variance, "descend");
% 
% plot(1:1000, sortedWeightedSumVar)
% 
% 
% [topVar, topVarIdx] = maxk(sum_weighted_variance, increase);
% 
% second_set_labels = reshape(label_blocks(:,topVarIdx,:),[],15);
% second_set_targets = reshape(target_blocks(:,topVarIdx,:), [], 1);
% 
% curr_set_labels = [first_set_labels; second_set_labels];
% curr_set_targets = [first_set_targets; second_set_targets];
% 
% % AL top with weighed variance
% cycles = 5;
% 
% weighted_ensembles = cell(cycles, numNets);
% variances = zeros(blockSize, numBlocks, cycles);
% mean_variances = zeros(numBlocks,cycles);
% sortedVarsArr = zeros(numBlocks,cycles);
% sortedVarsIdxArr = zeros(numBlocks,cycles);
% 
% 
% for k=1:5    
%     for i=1:numNets
% 
%         fprintf("AL, Cycle %d, Net %d", k, i)
% 
%         sz = size(curr_set_targets, 1);
%         shuf = randperm(sz);
% 
%         curr_labels = curr_set_labels(shuf,:);
%         curr_targets = curr_set_targets(shuf,:);
% 
%         szTrain = floor(0.9*sz);
%         szVal = floor(0.1*sz);
% 
%         currTrainLabels = curr_labels(1:szTrain,:);
%         currTrainTargets = curr_targets(1:szTrain,:);
% 
%         localOptions = mlpOptions;
%         localOptions.ValidationData = {curr_labels(szTrain+1:end,:), curr_targets(szTrain+1:end,:)};
% 
%         trainedNet = trainnet(currTrainLabels, currTrainTargets, mlpLayers,"mae", localOptions);
%         weighted_ensembles{k,i} = trainedNet;
%     end
% 
%     parfor i=1:numBlocks
%     if mod(i,50) == 0
%         fprintf("Simulation %d\n", i)
%     end
%         currSet = reshape(label_blocks(:,i,:), blockSize, 15);
%         results(:,i,:) = cell2mat(cellfun(@(net) predict(net,currSet), weighted_ensembles(k,:), "UniformOutput",false));
%     end
% 
% 
%     variances(:, :, k) = var(results, 0, 3);
%     weighed_variances = var_weights .* variances(:,:,k);
%     mean_variances(:,k) = sum(weighed_variances, 1)';
% 
%     [sortedVarsArr(:,k), sortedVarsIdxArr(:,k)] = sort(mean_variances(:,k), "descend");
% 
%     [topVar, topVarIdx] = maxk(mean_variances(:,k), increase);
% 
%     curr_set_labels = [curr_set_labels; reshape(label_blocks(:,topVarIdx,:),[],15)];
%     curr_set_targets = [curr_set_targets; reshape(target_blocks(:,topVarIdx,:), [], 1)];
% 
% end

%

comp_ensembles = cell(numCycles, numNets);
comp_variances = zeros(simBlockSize, numSims, numCycles);
comp_mean_variances = zeros(numSims,numCycles);
compSortedVarsArr = zeros(numSims,numCycles);
compSortedVarsIdxArr = zeros(numSims,numCycles);

compTrainSims = shuffled_simIDs(1:start+increase);
targetStats.random(1) = normalizeSpatialTargets(target_blocks(:,compTrainSims, :));

for k=1:numCycles
    for i=1:numNets

    fprintf("Comparison, Cycle %d, Net %d", k, i)
    
        sz = numel(compTrainSims);
        shuf = randperm(sz);
        
        shufIDs = compTrainSims(shuf);
        mu = targetStats.random(k).mu;
        sigma = targetStats.random(k).sigma;
    
        szTrain = floor(0.9*sz);
        szVal = floor(0.1*sz);
        
        currTrainIDs = shufIDs(1:szTrain);
        currValIDs = shufIDs(szTrain+1:end);
        
        currTrainLabels = reshape(label_blocks(:,currTrainIDs,:), [], nFeatures, 1);
        currTrainTargets = (reshape(target_blocks(:,currTrainIDs,:), [], 1) - mu) ./ sigma;
    
        currValLabels = reshape(label_blocks(:,currValIDs,:), [], nFeatures, 1);
        currValTargets = (reshape(target_blocks(:,currValIDs,:), [], 1) - mu) ./ sigma;
        
        localOptions = mlpOptions;
        localOptions.ValidationData = {currValLabels, currValTargets};

    trainedNet = trainnet(currTrainLabels, currTrainTargets, mlpLayers,"mae", localOptions);
    comp_ensembles{k,i} = trainedNet;

    end

    for i=1:numSims
    if mod(i,50) == 0
        fprintf("Simulation %d\n", i)
    end
        currSet = reshape(label_blocks(:,i,:), simBlockSize, nFeatures);
        results(:,i,:) = cell2mat(cellfun(@(net) predict(net,currSet), comp_ensembles(k,:), "UniformOutput",false));
    end

    comp_variances(:, :, k) = var(results, 0, 3);
    comp_mean_variances(:,k) = mean(comp_variances(:,:,k), 1);

    [compSortedVarsArr(:,k), compSortedVarsIdxArr(:,k)] = sort(comp_mean_variances(:,k), "descend");
    
    if k < numCycles
        compTrainSims = shuffled_simIDs(1:(start + (k+1)*increase));
        targetStats.random(k+1) = normalizeSpatialTargets(target_blocks(:,compTrainSims,:));
    end

end



% Hybrid method

combTrainSims = first_set_sims;
chosenIDs = selectHybridSims(mean_variance, combTrainSims, increase);

combTrainSims = [combTrainSims chosenIDs];

targetStats.alHyb(1) = normalizeSpatialTargets(target_blocks(:,combTrainSims, :));


comb_ensembles = cell(numCycles, numNets);
comb_variances = zeros(simBlockSize, numSims, numCycles);
comb_mean_variances = zeros(numSims,numCycles);
combSortedVarsArr = zeros(numSims,numCycles);
combSortedVarsIdxArr = zeros(numSims,numCycles);

for k=1:numCycles
    for i=1:numNets

    fprintf("Comparison, Cycle %d, Net %d", k, i)
    
    sz = numel(combTrainSims);
    shuf = randperm(sz);
    
    shufIDs = combTrainSims(shuf);
    mu = targetStats.alHyb(k).mu;
    sigma = targetStats.alHyb(k).sigma;

    szTrain = floor(0.9*sz);
    szVal = floor(0.1*sz);
    
    currTrainIDs = shufIDs(1:szTrain);
    currValIDs = shufIDs(szTrain+1:end);
    
    currTrainLabels = reshape(label_blocks(:,currTrainIDs,:), [], nFeatures, 1);
    currTrainTargets = (reshape(target_blocks(:,currTrainIDs,:), [], 1) - mu) ./ sigma;

    currValLabels = reshape(label_blocks(:,currValIDs,:), [], nFeatures, 1);
    currValTargets = (reshape(target_blocks(:,currValIDs,:), [], 1) - mu) ./ sigma;
    
    localOptions = mlpOptions;
    localOptions.ValidationData = {currValLabels, currValTargets};

    trainedNet = trainnet(currTrainLabels, currTrainTargets, mlpLayers,"mae", localOptions);
    comb_ensembles{k,i} = trainedNet;

    end

    for i=1:numSims
    if mod(i,50) == 0
        fprintf("Simulation %d\n", i)
    end
        currSet = reshape(label_blocks(:,i,:), simBlockSize, nFeatures);
        results(:,i,:) = cell2mat(cellfun(@(net) predict(net,currSet), comb_ensembles(k,:), "UniformOutput",false));
    end

    comb_variances(:, :, k) = var(results, 0, 3);
    comb_mean_variances(:,k) = mean(comb_variances(:,:,k), 1);

    [combSortedVarsArr(:,k), combSortedVarsIdxArr(:,k)] = sort(comb_mean_variances(:,k), "descend");
    
    if k < numCycles
        chosenIDs = selectHybridSims(comb_mean_variances(:,k), combTrainSims, increase);
        combTrainSims = [combTrainSims chosenIDs];
        targetStats.alHyb(k+1) = normalizeSpatialTargets(target_blocks(:,combTrainSims,:));
    end
end


close all;
% close all;
save(sprintf("Results/shen_temp_%d+5times%d_seed%d.mat", start, increase, seeds(currRun)));
end

%% Special case for weighted variance

% Predict using the shared initial ensemble.

weighted_results = zeros(simBlockSize, numSims, numNets);

for i=1:numSims
    if mod(i,50) == 0
        fprintf("Simulation %d\n", i)
    end

    currSet = reshape(label_blocks(:,i,:), simBlockSize, nFeatures);

    weighted_results(:,i,:) = reshape(cell2mat(cellfun( ...
        @(net) predict(net,currSet), ensemble, ...
        "UniformOutput",false)), simBlockSize, 1, numNets);
end

weighted_initial_variance = var(weighted_results, 0, 3);

% Fixed weights from all initially acquired temperature fields.
% Compute the initial ensemble's residuals in kelvin.

weighted_initial_predicted = ...
    mean(weighted_results(:,first_set_sims,:), 3) ...
    .* targetStats.initial.sigma + targetStats.initial.mu;

var_weights = mean(abs(weighted_initial_predicted ...
    - target_blocks(:,first_set_sims,:)), 2);

if max(var_weights) == 0
    var_weights = ones(simBlockSize, 1);
else
    var_weights = var_weights ./ max(var_weights);
end

weightSourceIDs = first_set_sims;

weighted_variance = var_weights .* weighted_initial_variance;
sum_weighted_variance = sum(weighted_variance, 1)';

[sortedWeightedSumVar, sortedWeightedSumVarIdx] = ...
    sort(sum_weighted_variance, "descend");

figure;
plot(1:numSims, sortedWeightedSumVar)

% Choose the first batch, excluding initially acquired simulations.

weightedTrainSims = first_set_sims(:).';

candidatePoolIDs = setdiff(1:numSims, weightedTrainSims);
[topVar, topVarIdx] = maxk( ...
    sum_weighted_variance(candidatePoolIDs), increase);

topVarIdx = candidatePoolIDs(topVarIdx);
weightedTrainSims = [weightedTrainSims topVarIdx(:).'];

targetStats.weighted(1) = ...
    normalizeSpatialTargets(target_blocks(:,weightedTrainSims,:));

weighted_ensembles = cell(numCycles, numNets);
weighted_variances = zeros(simBlockSize, numSims, numCycles);
weighted_mean_variances = zeros(numSims,numCycles);
weighted_sortedVarsArr = zeros(numSims,numCycles);
weighted_sortedVarsIdxArr = zeros(numSims,numCycles);

weightedHistory = cell(numCycles+1,1);
weightedHistory{1} = first_set_sims(:).';



for k=1:numCycles

    assert(numel(unique(weightedTrainSims)) == start + k*increase, ...
        "Incorrect weighted acquisition budget.");

    weightedHistory{k+1} = weightedTrainSims;

    for i=1:numNets

        fprintf("Weighted, Cycle %d, Net %d\n", k, i)

        sz = numel(weightedTrainSims);
        shuf = randperm(sz);

        shufIDs = weightedTrainSims(shuf);
        mu = targetStats.weighted(k).mu;
        sigma = targetStats.weighted(k).sigma;

        szTrain = floor(0.9*sz);

        currTrainIDs = shufIDs(1:szTrain);
        currValIDs = shufIDs(szTrain+1:end);

        currTrainLabels = reshape( ...
            label_blocks(:,currTrainIDs,:), [], nFeatures, 1);

        currTrainTargets = (reshape( ...
            target_blocks(:,currTrainIDs,:), [], 1) - mu) ./ sigma;

        currValLabels = reshape( ...
            label_blocks(:,currValIDs,:), [], nFeatures, 1);

        currValTargets = (reshape( ...
            target_blocks(:,currValIDs,:), [], 1) - mu) ./ sigma;

        localOptions = mlpOptions;
        localOptions.ValidationData = ...
            {currValLabels, currValTargets};

        trainedNet = trainnet(currTrainLabels, currTrainTargets, ...
            mlpLayers,"mae",localOptions);

        weighted_ensembles{k,i} = trainedNet;
    end

    for i=1:numSims
        if mod(i,50) == 0
            fprintf("Simulation %d\n", i)
        end

        currSet = reshape(label_blocks(:,i,:), ...
            simBlockSize, nFeatures);

        weighted_results(:,i,:) = reshape(cell2mat(cellfun( ...
            @(net) predict(net,currSet), weighted_ensembles(k,:), ...
            "UniformOutput",false)), simBlockSize, 1, numNets);
    end

    weighted_variances(:,:,k) = var(weighted_results, 0, 3);

    weighed_variances = var_weights .* weighted_variances(:,:,k);
    weighted_mean_variances(:,k) = sum(weighed_variances, 1)';

    [weighted_sortedVarsArr(:,k), weighted_sortedVarsIdxArr(:,k)] = ...
        sort(weighted_mean_variances(:,k), "descend");

    if k < numCycles

        candidatePoolIDs = setdiff(1:numSims, weightedTrainSims);

        [topVar, topVarIdx] = maxk( ...
            weighted_mean_variances(candidatePoolIDs,k), increase);

        topVarIdx = candidatePoolIDs(topVarIdx);
        weightedTrainSims = [weightedTrainSims topVarIdx(:).'];

        targetStats.weighted(k+1) = normalizeSpatialTargets(target_blocks(:,weightedTrainSims,:));
    end
end


%% QBC: 99th percentile of pointwise ensemble range

% Predict using the same shared initial ensemble.

qbc_results = zeros(simBlockSize, numSims, numNets);

for i=1:numSims
    if mod(i,50) == 0
        fprintf("Simulation %d\n", i)
    end

    currSet = reshape(label_blocks(:,i,:), simBlockSize, nFeatures);

    qbc_results(:,i,:) = reshape(cell2mat(cellfun( ...
        @(net) predict(net,currSet), ensemble, ...
        "UniformOutput",false)), simBlockSize, 1, numNets);
end


maxDiff = max(qbc_results, [], 3) - min(qbc_results, [], 3);
maxDiff99 = prctile(maxDiff,99,1)';

[qbc_sortedVars, qbc_sortedVarsIdx] = sort(maxDiff99, "descend");

figure;
plot(1:numSims, qbc_sortedVars)

% Choose the first batch, excluding initially acquired simulations.

qbcTrainSims = first_set_sims(:).';

candidatePoolIDs = setdiff(1:numSims, qbcTrainSims);
[topVar, topVarIdx] = maxk(maxDiff99(candidatePoolIDs), increase);

topVarIdx = candidatePoolIDs(topVarIdx);
qbcTrainSims = [qbcTrainSims topVarIdx(:).'];

targetStats.qbc(1) = normalizeSpatialTargets(target_blocks(:,qbcTrainSims,:));

qbc_ensembles = cell(numCycles, numNets);
maxDiff = zeros(simBlockSize, numSims, numCycles);
maxDiff99 = zeros(numSims,numCycles);
qbc_sortedVarsArr2 = zeros(numSims,numCycles);
qbc_sortedVarsIdxArr2 = zeros(numSims,numCycles);

qbcHistory = cell(numCycles+1,1);
qbcHistory{1} = first_set_sims(:).';

for k=1:numCycles

    assert(numel(unique(qbcTrainSims)) == start + k*increase, ...
        "Incorrect QBC acquisition budget.");

    qbcHistory{k+1} = qbcTrainSims;

    for i=1:numNets

        fprintf("QBC, Cycle %d, Net %d\n", k, i)

        sz = numel(qbcTrainSims);
        shuf = randperm(sz);

        shufIDs = qbcTrainSims(shuf);
        mu = targetStats.qbc(k).mu;
        sigma = targetStats.qbc(k).sigma;

        szTrain = floor(0.9*sz);

        currTrainIDs = shufIDs(1:szTrain);
        currValIDs = shufIDs(szTrain+1:end);

        currTrainLabels = reshape(label_blocks(:,currTrainIDs,:), [], nFeatures, 1);
        currTrainTargets = (reshape(target_blocks(:,currTrainIDs,:), [], 1) - mu) ./ sigma;

        currValLabels = reshape(label_blocks(:,currValIDs,:), [], nFeatures, 1);
        currValTargets = (reshape(target_blocks(:,currValIDs,:), [], 1) - mu) ./ sigma;

        localOptions = mlpOptions;
        localOptions.ValidationData = ...
            {currValLabels, currValTargets};

        trainedNet = trainnet(currTrainLabels, currTrainTargets,mlpLayers,"mae",localOptions);

        qbc_ensembles{k,i} = trainedNet;
    end

    for i=1:numSims
        if mod(i,50) == 0
            fprintf("Simulation %d\n", i)
        end

        currSet = reshape(label_blocks(:,i,:),simBlockSize, nFeatures);

        qbc_results(:,i,:) = reshape(cell2mat(cellfun(@(net) predict(net,currSet), qbc_ensembles(k,:),"UniformOutput",false)), simBlockSize, 1, numNets);
    end

    maxDiff(:,:,k) = max(qbc_results, [], 3) - min(qbc_results, [], 3);

    maxDiff99(:,k) = prctile(maxDiff(:,:,k),99,1)';

    [qbc_sortedVarsArr2(:,k), qbc_sortedVarsIdxArr2(:,k)] = sort(maxDiff99(:,k), "descend");

    if k < numCycles

        candidatePoolIDs = setdiff(1:numSims, qbcTrainSims);

        [topVar, topVarIdx] = maxk(maxDiff99(candidatePoolIDs,k), increase);

        topVarIdx = candidatePoolIDs(topVarIdx);
        qbcTrainSims = [qbcTrainSims topVarIdx(:).'];

        targetStats.qbc(k+1) = normalizeSpatialTargets(target_blocks(:,qbcTrainSims,:));
    end
end

save("Results/qbc_ensembles.mat", "qbc_ensembles", "targetStats")
save("Results/weighted_ensembles.mat", "weighted_ensembles", "targetStats")
