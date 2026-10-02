starts = [10, 20, 30, 40, 50];
increases = [4, 8, 10, 20, 30];

seeds = load("Data/seeds.mat").seeds;

for currRun=1:5

clearvars -except starts increases seeds currRun

labels = load("shenData5.mat").labels;
targets = load("shenData5.mat").targets;

numData = size(labels, 1);

load("Data\training_normalization.mat")

normalized_labels = (labels - labels_mean) ./ labels_std;
normalized_targets = (targets - targets_mean) ./ targets_std;

% currSeed = 1234;
rng(seeds(currRun))
% rng(currSeed);
fullShuf = randperm(size(targets, 1));

shuffled_labels = normalized_labels(fullShuf,:);
shuffled_targets = normalized_targets(fullShuf,:);

% currRun = 4;

start = starts(4);
increase = increases(4);

first_set_labels = shuffled_labels(1:start,:);
first_set_targets = shuffled_targets(1:start, :);

% Network

mlpLayers = [
    featureInputLayer(6,'Normalization','None')
    fullyConnectedLayer(100)
    reluLayer
    fullyConnectedLayer(100)
    reluLayer
    fullyConnectedLayer(1)
    identityLayer];

options = trainingOptions('adam', ...
    'InitialLearnRate',1e-3, ...
    'MiniBatchSize', 128, ...
    'Verbose', true,  ...
    "VerboseFrequency", 250, ...
    'Shuffle','every-epoch', ...
    'ExecutionEnvironment','gpu',...
    'ValidationFrequency',250,...
    "ValidationPatience",10);

mlpOptions = options;
mlpOptions.MaxEpochs = 10000;

% Base set

tic;

numNets = 30;
ensemble = cell(1, numNets);

for i=1:numNets

    fprintf("Curr net: %d\n", i)
    
    sz = size(first_set_targets, 1);
    shuf = randperm(sz);
    
    curr_labels = first_set_labels(shuf,:);
    curr_targets = first_set_targets(shuf,:);

    szTrain = floor(0.9*sz);
    szVal = floor(0.1*sz);
    
    currTrainLabels = curr_labels(1:szTrain,:);
    currTrainTargets = curr_targets(1:szTrain,:);

    currValLabels = curr_labels(szTrain+1:end, :);
    currValTargets = curr_targets(szTrain+1:end, :);

    mlpOptions.ValidationData = {currValLabels, currValTargets};

    trainedNet = trainnet(currTrainLabels, currTrainTargets, mlpLayers,"mae", mlpOptions);
    ensemble{i} = trainedNet;

end

result  = cell2mat(cellfun(@(net) predict(net,normalized_labels),ensemble, "UniformOutput", false));

variance = var(result,[],2);

[sortedVars, sortedVarsIdx] = sort(variance, "descend");

initial_ensemble_time = toc;


alTrainIdx = fullShuf(1:40);
candidatePoolIdx = setdiff(1:numel(targets), alTrainIdx);
[~, alCandidatePoolVarianceIdx] = sort(variance(candidatePoolIdx, :), "descend");
alSelectedIdx = candidatePoolIdx(alCandidatePoolVarianceIdx(1:increase));

alTrainIdx = [alTrainIdx alSelectedIdx];

al_labels = normalized_labels(alTrainIdx, :);
al_targets = normalized_targets(alTrainIdx, :);

numNets = 30;
numCycles = 5;
numData = size(targets, 1);

al_ensembles = cell(numCycles, numNets);
al_variances = zeros(size(targets, 1),numCycles);



tic;
for cycle=5:numCycles
    
    disp(cycle)

    for i=1:numNets
        
        fprintf("AL: Curr cycle: %d - Curr net: %d", cycle, i)
        sz = size(al_targets, 1);
        shuf = randperm(sz);
        
        al_labels = al_labels(shuf,:);
        al_targets = al_targets(shuf, :);
    
        szTrain = floor(0.9*sz);
        szVal = floor(0.1*sz);
        
        currTrainLabels = al_labels(1:szTrain,:);
        currTrainTargets = al_targets(1:szTrain,:);

        currValLabels = al_labels(szTrain+1:end, :);
        currValTargets = al_targets(szTrain+1:end, :);

        mlpOptions.ValidationData = {currValLabels, currValTargets};
    
        trainedNet = trainnet(currTrainLabels, currTrainTargets, mlpLayers,"mae", mlpOptions);
        al_ensembles{cycle,i} = trainedNet;
    end

    results = cell2mat(cellfun(@(net) predict(net, normalized_labels), al_ensembles(cycle, :), "UniformOutput",false));
    al_variances(:,cycle) = var(results,[],2);

    candidatePoolIdx = setdiff(1:numel(targets), alTrainIdx);
    [~, alCandidatePoolVarianceIdx] = sort(al_variances(candidatePoolIdx, cycle), "descend");
    alSelectedIdx = candidatePoolIdx(alCandidatePoolVarianceIdx(1:increase));
    
    alTrainIdx = [alTrainIdx alSelectedIdx];
    
    al_labels = normalized_labels(alTrainIdx, :);
    al_targets = normalized_targets(alTrainIdx, :);

end

last_ensemble_time = toc;
%
time1 = toc;

% Hybrid approach

combTrainIdx = fullShuf(1:start);
candidatePoolIdx = setdiff(1:numel(targets), combTrainIdx);
[~, combCandidatePoolVarianceIdx] = sort(variance(candidatePoolIdx, 1), "descend");

topIdx = candidatePoolIdx(combCandidatePoolVarianceIdx(1:increase/2));

sz = numel(candidatePoolIdx);

equiLocal = linspace(increase/2 + 1, sz, increase/2);
equiLocal = floor(equiLocal);

equiIdx = candidatePoolIdx(combCandidatePoolVarianceIdx(equiLocal));

combTrainIdx = [combTrainIdx topIdx equiIdx];

combinationLabels = normalized_labels(combTrainIdx, :);
combinationTargets = normalized_targets(combTrainIdx, :);

combination_ensembles = cell(numCycles, numNets);
combination_variances = zeros(size(targets, 1),numCycles);


for cycle=1:numCycles
    
    for i=1:numNets
        
        fprintf("Hyb: Curr cycle: %d - Curr net: %d\n", cycle, i)
        sz = size(combinationTargets, 1);
        shuf = randperm(sz);
        
        combinationLabels = combinationLabels(shuf,:);
        combinationTargets = combinationTargets(shuf, :);
    
        szTrain = floor(0.9*sz);
        szVal = floor(0.1*sz);
        
        currTrainLabels = combinationLabels(1:szTrain,:);
        currTrainTargets = combinationTargets(1:szTrain,:);

        currValLabels = combinationLabels(szTrain+1:end, :);
        currValTargets = combinationTargets(szTrain+1:end, :);

        mlpOptions.ValidationData = {currValLabels, currValTargets};
    
        trainedNet = trainnet(currTrainLabels, currTrainTargets, mlpLayers,"mae", mlpOptions);
        combination_ensembles{cycle,i} = trainedNet;
    end

    results = cell2mat(cellfun(@(net) predict(net, normalized_labels), combination_ensembles(cycle, :), "UniformOutput",false));
    combination_variances(:,cycle) = var(results,[],2);

    candidatePoolIdx = setdiff(1:numel(targets), combTrainIdx);
    [~, combCandidatePoolVarianceIdx] = sort(combination_variances(candidatePoolIdx, cycle), "descend");
    
    topIdx = candidatePoolIdx(combCandidatePoolVarianceIdx(1:increase/2));
    
    sz = numel(candidatePoolIdx);
    
    equiLocal = linspace(increase/2 + 1, sz, increase/2);
    equiLocal = floor(equiLocal);
    
    equiIdx = candidatePoolIdx(combCandidatePoolVarianceIdx(equiLocal));
    
    combTrainIdx = [combTrainIdx topIdx equiIdx];
    
    combinationLabels = normalized_labels(combTrainIdx, :);
    combinationTargets = normalized_targets(combTrainIdx, :);

end

% Comparison

comparison_ensembles = cell(numCycles, numNets);
comparison_variances = zeros(size(targets, 1),numCycles);


for cycle=1:numCycles
    
    comparisonLabels = shuffled_labels(1:start+cycle*increase, :);
    comparisonTargets = shuffled_targets(1:start+cycle*increase, :);


    for i=1:numNets
        
        fprintf("Random: Curr cycle: %d - Curr net: %d\n", cycle, i)
        sz = size(comparisonTargets, 1);
        shuf = randperm(sz);
        
        comparisonLabels = comparisonLabels(shuf,:);
        comparisonTargets = comparisonTargets(shuf, :);
    
        szTrain = floor(0.9*sz);
        szVal = floor(0.1*sz);
        
        currTrainLabels = comparisonLabels(1:szTrain,:);
        currTrainTargets = comparisonTargets(1:szTrain,:);

        currValLabels = comparisonLabels(szTrain+1:end, :);
        currValTargets = comparisonTargets(szTrain+1:end, :);

        mlpOptions.ValidationData = {currValLabels, currValTargets};

        trainedNet = trainnet(currTrainLabels, currTrainTargets, mlpLayers,"mae", mlpOptions);
        comparison_ensembles{cycle,i} = trainedNet;
    end

    results = cell2mat(cellfun(@(net) predict(net, normalized_labels), comparison_ensembles(cycle, :), "UniformOutput",false));
    comparison_variances(:,cycle) = var(results,[],2);

end


% close all;
save(sprintf("Results/IV_AL_cycles_%d+5times%d_seed%d_shenData5.mat", starts(4), increases(4), seeds(currRun)));
% 
% end