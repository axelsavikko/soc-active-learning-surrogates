
seeds = load("Data/seeds.mat").seeds;

for currRun=1:5

rng(seeds(currRun));

% Data load
clear *
labels = load("Data/spatial_candidate_samples.mat").labels;
targets = load("Data/spatial_candidate_samples.mat").targets;

load("Data/training_normalization_spatial.mat")

normalized_labels = (labels - labels_mean) ./ labels_std;
normalized_targets = (targets - targets_mean) ./ targets_std;

% Data

normalized_targets = normalized_targets(:,2);

shuf = randperm(size(normalized_targets, 1));

shuffled_labels = normalized_labels(shuf,:);
shuffled_targets = normalized_targets(shuf,:);

% Making first currSet as 3 random full simulations


blockSize = 28*28;
numBlocks = floor(length(normalized_targets) / blockSize);

% Reshape into blocks (each column is a block)
label_blocks = reshape(normalized_labels(1:numBlocks * blockSize, :), blockSize, numBlocks, []);
target_blocks = reshape(normalized_targets(1:numBlocks * blockSize, :), blockSize, numBlocks, []);

% Reshape unnormalized data into blocks (each column is a block)
label_blocks2 = reshape(labels(1:numBlocks * blockSize, :), blockSize, numBlocks, []);
target_blocks2 = reshape(targets(1:numBlocks * blockSize, :), blockSize, numBlocks, []);

% label_blocks = label_blocks(:,1:2:end, :);
% target_blocks = target_blocks(:, 1:2:end, :);

numBlocks = size(label_blocks, 2);

% Shuffle the blocks
shuffledIndices = randperm(numBlocks);
shuffledLabelBlocks = label_blocks(:,shuffledIndices, :);
shuffledTargetBlocks = target_blocks(:,shuffledIndices, :);

% Flatten back into a single array
shuffled_labels = reshape(shuffledLabelBlocks, [], size(normalized_labels, 2), 1);
shuffled_targets = reshape(shuffledTargetBlocks, [], size(normalized_targets, 2), 1);


% First set
start = 40;
oneSet = blockSize;

first_set_labels = shuffled_labels(1:start*oneSet,:);
first_set_targets = shuffled_targets(1:start*oneSet,:);



% Comparison

increase = 20;

comparison_set_labels = shuffled_labels(1:start*oneSet+5*increase*oneSet, :);
comparison_set_targets = shuffled_targets(1:start*oneSet+5*increase*oneSet, :);

comparison_set_labels2 = shuffled_labels(1:start*oneSet+increase*2*oneSet, :);
comparison_set_targets2 = shuffled_targets(1:start*oneSet+increase*2*oneSet, :);


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

ensemble = cell(1, numNets);


for i=1:numNets

    fprintf("Base Ensemble, Net %d\n", i)
    sz = size(first_set_targets, 1);
    shuf = randperm(sz);
    
    curr_labels = first_set_labels(shuf,:);
    curr_targets = first_set_targets(shuf,:);

    szTrain = floor(0.9*sz);
    szVal = floor(0.1*sz);
    
    currTrainLabels = curr_labels(1:szTrain,:);
    currTrainTargets = curr_targets(1:szTrain,:);
    
    localOptions = mlpOptions;
    localOptions.ValidationData = {curr_labels(szTrain+1:end,:), curr_targets(szTrain+1:end,:)};

    trainedNet = trainnet(currTrainLabels, currTrainTargets, mlpLayers,"mae", localOptions);
    ensemble{i} = trainedNet;

end
first_net_time = toc;
% Predicting for all labels with all networks:

results = zeros(blockSize, numBlocks, numNets);

parfor i=1:numBlocks
    if mod(i,50) == 0
        fprintf("Simulation %d\n", i)
    end
    currSet = reshape(label_blocks(:,i,:), blockSize, 15);
    results(:,i,:) = cell2mat(cellfun(@(net) predict(net,currSet), ensemble, "UniformOutput",false));
end

variance = var(results, 0, 3);
% sum_variance = sum(variance,1)';
% mean_variance = mean(variance, 1)';

%

mean_variance = mean(variance, 1)';
[sortedVars, sortedVarsIdx] = sort(mean_variance, "descend");
plot(1:numBlocks, sortedVars)

% Choose 20 top variance

[topVar, topVarIdx] = maxk(mean_variance, increase);

second_set_labels = reshape(label_blocks(:,topVarIdx,:),[],15);
second_set_targets = reshape(target_blocks(:,topVarIdx,:), [], 1);

curr_set_labels = [first_set_labels; second_set_labels];
curr_set_targets = [first_set_targets; second_set_targets];

%
cycles = 5;

al_ensembles = cell(cycles, numNets);
variances = zeros(blockSize, numBlocks, cycles);
mean_variances = zeros(numBlocks,cycles);
sortedVarsArr = zeros(numBlocks,cycles);
sortedVarsIdxArr = zeros(numBlocks,cycles);

for k=1:5    
    for i=1:numNets

        fprintf("AL, Cycle %d, Net %d", k, i)
        
        sz = size(curr_set_targets, 1);
        shuf = randperm(sz);
        
        curr_labels = curr_set_labels(shuf,:);
        curr_targets = curr_set_targets(shuf,:);
    
        szTrain = floor(0.9*sz);
        szVal = floor(0.1*sz);
        
        currTrainLabels = curr_labels(1:szTrain,:);
        currTrainTargets = curr_targets(1:szTrain,:);
        
        localOptions = mlpOptions;
        localOptions.ValidationData = {curr_labels(szTrain+1:end,:), curr_targets(szTrain+1:end,:)};
    
        trainedNet = trainnet(currTrainLabels, currTrainTargets, mlpLayers,"mae", localOptions);
        al_ensembles{k,i} = trainedNet;
    end

    parfor i=1:numBlocks
    if mod(i,50) == 0
        fprintf("Simulation %d\n", i)
    end
        currSet = reshape(label_blocks(:,i,:), blockSize, 15);
        results(:,i,:) = cell2mat(cellfun(@(net) predict(net,currSet), al_ensembles(k,:), "UniformOutput",false));
    end


    variances(:, :, k) = var(results, 0, 3);
    mean_variances(:,k) = mean(variances(:,:,k), 1);

    [sortedVarsArr(:,k), sortedVarsIdxArr(:,k)] = sort(mean_variances(:,k), "descend");

    [topVar, topVarIdx] = maxk(mean_variances(:,k), increase);
    
    curr_set_labels = [curr_set_labels; reshape(label_blocks(:,topVarIdx,:),[],15)];
    curr_set_targets = [curr_set_targets; reshape(target_blocks(:,topVarIdx,:), [], 1)];

end


%

comp_ensembles = cell(cycles, numNets);
comp_variances = zeros(blockSize, numBlocks, cycles);
comp_mean_variances = zeros(numBlocks,cycles);
compSortedVarsArr = zeros(numBlocks,cycles);
compSortedVarsIdxArr = zeros(numBlocks,cycles);

compSetLabels = shuffled_labels(1:start*oneSet+increase*oneSet, :);
compSetTargets = shuffled_targets(1:start*oneSet+increase*oneSet, :);

for k=1:5    
    for i=1:numNets

    fprintf("Comparison, Cycle %d, Net %d", k, i)
    
    sz = size(compSetLabels, 1);
    shuf = randperm(sz);
    
    curr_labels = compSetLabels(shuf,:);
    curr_targets = compSetTargets(shuf,:);

    szTrain = floor(0.9*sz);
    szVal = floor(0.1*sz);
    
    currTrainLabels = curr_labels(1:szTrain,:);
    currTrainTargets = curr_targets(1:szTrain,:);
    
    localOptions = mlpOptions;
    localOptions.ValidationData = {curr_labels(szTrain+1:end,:), curr_targets(szTrain+1:end,:)};

    trainedNet = trainnet(currTrainLabels, currTrainTargets, mlpLayers,"mae", localOptions);
    comp_ensembles{k,i} = trainedNet;

    end

    parfor i=1:numBlocks
    if mod(i,50) == 0
        fprintf("Simulation %d\n", i)
    end
        currSet = reshape(label_blocks(:,i,:), blockSize, 15);
        results(:,i,:) = cell2mat(cellfun(@(net) predict(net,currSet), comp_ensembles(k,:), "UniformOutput",false));
    end

    comp_variances(:, :, k) = var(results, 0, 3);
    comp_mean_variances(:,k) = mean(comp_variances(:,:,k), 1);

    [compSortedVarsArr(:,k), compSortedVarsIdxArr(:,k)] = sort(comp_mean_variances(:,k), "descend");
    
    compSetLabels = shuffled_labels(1:start*oneSet+(k+1)*increase*oneSet, :);
    compSetTargets = shuffled_targets(1:start*oneSet+(k+1)*increase*oneSet, :);

end

%

notIncluded = normalized_labels(~ismember(normalized_labels, first_set_labels, "rows"));
notIncluded = reshape(notIncluded, blockSize, [], 1);
sz = size(notIncluded,2);
equiIdxs = linspace((increase/2)+1,sz, increase/2);
equiIdxs = floor(equiIdxs);
equiIdxs = sortedVarsIdx(equiIdxs);

combSetLabels = [first_set_labels ; reshape(label_blocks(:,sortedVarsIdx(1:increase/2),:), [], 15) ; reshape(label_blocks(:,equiIdxs,:), [], 15)];
combSetTargets = [first_set_targets ;  reshape(target_blocks(:,sortedVarsIdx(1:increase/2),:), [], 1) ; reshape(target_blocks(:,equiIdxs,:), [], 1)];

% Hybrid method

comb_ensembles = cell(cycles, numNets);
comb_variances = zeros(blockSize, numBlocks, cycles);
comb_mean_variances = zeros(numBlocks,cycles);
combSortedVarsArr = zeros(numBlocks,cycles);
combSortedVarsIdxArr = zeros(numBlocks,cycles);

for k=1:5    
    for i=1:numNets

    fprintf("Comparison, Cycle %d, Net %d", k, i)
    
    sz = size(combSetLabels, 1);
    shuf = randperm(sz);
    
    curr_labels = combSetLabels(shuf,:);
    curr_targets = combSetTargets(shuf,:);

    szTrain = floor(0.9*sz);
    szVal = floor(0.1*sz);
    
    currTrainLabels = curr_labels(1:szTrain,:);
    currTrainTargets = curr_targets(1:szTrain,:);
    
    localOptions = mlpOptions;
    localOptions.ValidationData = {curr_labels(szTrain+1:end,:), curr_targets(szTrain+1:end,:)};

    trainedNet = trainnet(currTrainLabels, currTrainTargets, mlpLayers,"mae", localOptions);
    comb_ensembles{k,i} = trainedNet;

    end

    parfor i=1:numBlocks
    if mod(i,50) == 0
        fprintf("Simulation %d\n", i)
    end
        currSet = reshape(label_blocks(:,i,:), blockSize, 15);
        results(:,i,:) = cell2mat(cellfun(@(net) predict(net,currSet), comb_ensembles(k,:), "UniformOutput",false));
    end

    comb_variances(:, :, k) = var(results, 0, 3);
    comb_mean_variances(:,k) = mean(comb_variances(:,:,k), 1);

    [combSortedVarsArr(:,k), combSortedVarsIdxArr(:,k)] = sort(comb_mean_variances(:,k), "descend");
    
    notIncluded = normalized_labels(~ismember(normalized_labels, combSetLabels, "rows"));
    notIncluded = reshape(notIncluded, blockSize, [], 1);
    sz = size(notIncluded,2);
    equiIdxs = linspace((increase/2)+1,sz, increase/2);
    equiIdxs = floor(equiIdxs);
    equiIdxs = combSortedVarsIdxArr(equiIdxs,k);
    
    combSetLabels = [combSetLabels ; reshape(label_blocks(:,combSortedVarsIdxArr(1:increase/2,k), :), [], 15) ; reshape(label_blocks(:,equiIdxs,:), [], 15)];
    combSetTargets = [combSetTargets ;  reshape(target_blocks(:,combSortedVarsIdxArr(1:increase/2,k),:), [], 1) ; reshape(target_blocks(:,equiIdxs,:), [], 1)];

end


close all;
save(sprintf("shen_temp_%d+5times%d_seed%d.mat", start, increase, seeds(currRun)))
end



%% Special case for weighted variance and QBC

numNets = 30;

ensemble = cell(1, numNets);

for i=1:numNets

    fprintf("Base Ensemble, Net %d\n", i)
    sz = size(first_set_targets, 1);
    shuf = randperm(sz);
    
    curr_labels = first_set_labels(shuf,:);
    curr_targets = first_set_targets(shuf,:);

    szTrain = floor(0.9*sz);
    szVal = floor(0.1*sz);
    
    currTrainLabels = curr_labels(1:szTrain,:);
    currTrainTargets = curr_targets(1:szTrain,:);
    
    localOptions = mlpOptions;
    localOptions.ValidationData = {curr_labels(szTrain+1:end,:), curr_targets(szTrain+1:end,:)};

    trainedNet = trainnet(currTrainLabels, currTrainTargets, mlpLayers,"mae", localOptions);
    ensemble{i} = trainedNet;

end

% Predicting for all labels with all networks:

results = zeros(blockSize, numBlocks, numNets);

for i=1:numBlocks
    if mod(i,50) == 0
        fprintf("Simulation %d\n", i)
    end
    currSet = reshape(label_blocks(:,i,:), blockSize, 15);
    results(:,i,:) = cell2mat(cellfun(@(net) predict(net,currSet), ensemble, "UniformOutput",false));
end

variance = var(results, 0, 3);

% Choose 20 points based on weighted variances

var_weights = load("Data/var_weights_5.mat").var_weights_5;
var_weights = reshape(var_weights, 28*28, []);

weighted_variance = var_weights .* variance;
sum_weighted_variance = sum(weighted_variance, 1)';

[sortedWeightedSumVar, sortedWeightedSumVarIdx] = sort(sum_weighted_variance, "descend");

plot(1:1000, sortedWeightedSumVar)


[topVar, topVarIdx] = maxk(sum_weighted_variance, increase);

second_set_labels = reshape(label_blocks(:,topVarIdx,:),[],15);
second_set_targets = reshape(target_blocks(:,topVarIdx,:), [], 1);

curr_set_labels = [first_set_labels; second_set_labels];
curr_set_targets = [first_set_targets; second_set_targets];

% AL top with weighed variance
cycles = 5;

weighted_ensembles = cell(cycles, numNets);
variances = zeros(blockSize, numBlocks, cycles);
mean_variances = zeros(numBlocks,cycles);
sortedVarsArr = zeros(numBlocks,cycles);
sortedVarsIdxArr = zeros(numBlocks,cycles);


for k=1:5    
    for i=1:numNets

        fprintf("AL, Cycle %d, Net %d", k, i)

        sz = size(curr_set_targets, 1);
        shuf = randperm(sz);

        curr_labels = curr_set_labels(shuf,:);
        curr_targets = curr_set_targets(shuf,:);

        szTrain = floor(0.9*sz);
        szVal = floor(0.1*sz);

        currTrainLabels = curr_labels(1:szTrain,:);
        currTrainTargets = curr_targets(1:szTrain,:);

        localOptions = mlpOptions;
        localOptions.ValidationData = {curr_labels(szTrain+1:end,:), curr_targets(szTrain+1:end,:)};

        trainedNet = trainnet(currTrainLabels, currTrainTargets, mlpLayers,"mae", localOptions);
        weighted_ensembles{k,i} = trainedNet;
    end

    for i=1:numBlocks
    if mod(i,50) == 0
        fprintf("Simulation %d\n", i)
    end
        currSet = reshape(label_blocks(:,i,:), blockSize, 15);
        results(:,i,:) = cell2mat(cellfun(@(net) predict(net,currSet), weighted_ensembles(k,:), "UniformOutput",false));
    end


    variances(:, :, k) = var(results, 0, 3);
    weighed_variances = var_weights .* variances(:,:,k);
    mean_variances(:,k) = sum(weighed_variances, 1)';

    [sortedVarsArr(:,k), sortedVarsIdxArr(:,k)] = sort(mean_variances(:,k), "descend");

    [topVar, topVarIdx] = maxk(mean_variances(:,k), increase);

    curr_set_labels = [curr_set_labels; reshape(label_blocks(:,topVarIdx,:),[],15)];
    curr_set_targets = [curr_set_targets; reshape(target_blocks(:,topVarIdx,:), [], 1)];

end


%% QBC


numNets = 30;

ensemble = cell(1, numNets);

for i=1:numNets

    fprintf("Base Ensemble, Net %d\n", i)
    sz = size(first_set_targets, 1);
    shuf = randperm(sz);
    
    curr_labels = first_set_labels(shuf,:);
    curr_targets = first_set_targets(shuf,:);

    szTrain = floor(0.9*sz);
    szVal = floor(0.1*sz);
    
    currTrainLabels = curr_labels(1:szTrain,:);
    currTrainTargets = curr_targets(1:szTrain,:);
    
    localOptions = mlpOptions;
    localOptions.ValidationData = {curr_labels(szTrain+1:end,:), curr_targets(szTrain+1:end,:)};

    trainedNet = trainnet(currTrainLabels, currTrainTargets, mlpLayers,"mae", localOptions);
    ensemble{i} = trainedNet;

end

% Predicting for all labels with all networks:

results = zeros(blockSize, numBlocks, numNets);

for i=1:numBlocks
    if mod(i,50) == 0
        fprintf("Simulation %d\n", i)
    end
    currSet = reshape(label_blocks(:,i,:), blockSize, 15);
    results(:,i,:) = cell2mat(cellfun(@(net) predict(net,currSet), ensemble, "UniformOutput",false));
end

variance = var(results, 0, 3);
%%
maxDiff = max(results, [], 3) - min(results, [], 3);
maxDiff95 = prctile(maxDiff ,99, 1);
maxDiff95mean = mean(maxDiff95, 2);

[sortedVars, sortedVarsIdx] = sort(maxDiff95, "descend");
plot(1:numBlocks, sortedVars)


% Choose 20 top variance

[topVar, topVarIdx] = maxk(maxDiff95, increase);

second_set_labels = reshape(label_blocks(:,topVarIdx,:),[],15);
second_set_targets = reshape(target_blocks(:,topVarIdx,:), [], 1);

curr_set_labels = [first_set_labels; second_set_labels];
curr_set_targets = [first_set_targets; second_set_targets];

%
cycles = 5;

al_ensembles = cell(cycles, numNets);
maxDiff = zeros(blockSize, numBlocks, cycles);
maxDiff95 = zeros(numBlocks,cycles);
sortedVarsArr = zeros(numBlocks,cycles);
sortedVarsIdxArr = zeros(numBlocks,cycles);



for k=1:cycles    
    for i=1:numNets

        fprintf("AL, Cycle %d, Net %d", k, i)
        
        sz = size(curr_set_targets, 1);
        shuf = randperm(sz);
        
        curr_labels = curr_set_labels(shuf,:);
        curr_targets = curr_set_targets(shuf,:);
    
        szTrain = floor(0.9*sz);
        szVal = floor(0.1*sz);
        
        currTrainLabels = curr_labels(1:szTrain,:);
        currTrainTargets = curr_targets(1:szTrain,:);
        
        localOptions = mlpOptions;
        localOptions.ValidationData = {curr_labels(szTrain+1:end,:), curr_targets(szTrain+1:end,:)};
    
        trainedNet = trainnet(currTrainLabels, currTrainTargets, mlpLayers,"mae", localOptions);
        al_ensembles{k,i} = trainedNet;
    end

    for i=1:numBlocks
    if mod(i,50) == 0
        fprintf("Simulation %d\n", i)
    end
        currSet = reshape(label_blocks(:,i,:), blockSize, 15);
        results(:,i,:) = cell2mat(cellfun(@(net) predict(net,currSet), al_ensembles(k,:), "UniformOutput",false));
    end


    maxDiff(:, :, k) = max(results, [], 3) - min(results, [], 3);
    maxDiff95(:,k) = prctile(maxDiff(:,:,k),99, 1);

    [sortedVarsArr2(:,k), sortedVarsIdxArr2(:,k)] = sort(maxDiff95(:,k), "descend");

    [topVar, topVarIdx] = maxk(maxDiff95(:,k), increase);
    
    curr_set_labels = [curr_set_labels; reshape(label_blocks(:,topVarIdx,:),[],15)];
    curr_set_targets = [curr_set_targets; reshape(target_blocks(:,topVarIdx,:), [], 1)];

end
