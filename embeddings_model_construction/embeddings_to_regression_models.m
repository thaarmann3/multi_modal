%% STEP 3 (MATLAB): Ridge + PLS2 + Kernel Ridge (RBF) + Neural Net (MLP) with selectable raters
clear; clc; close all;
rng(1);
if isempty(gcp('nocreate'))
    parpool;   % uses default number of workers
end

%% ===== USER EDITS =====
labelCsv = "sbert_potentialfield_commands_2000_sentences_3_raters_learnable.csv";
embCsv   = "embeddings_384.csv";   % can be embeddings_384.csv or embeddings_768.csv

% Select which raters to train/evaluate on:
% Examples: [1], [1 2], [2 3], [1 2 3]
raterSelection = [1 2 3];

K_outer = 5;               % outer CV folds
K_inner = 5;               % inner CV folds (for hyperparameter selection)

% Ridge hyperparameter grid
lambdaGrid = logspace(-6, 6, 25);

% PLS hyperparameter grid
maxPLSComp = 1; %90;

% ---- Kernel Ridge (RBF) hyperparameter grids ----
% sigma is in standardized-X units (since we zscore X each fold)
tuneRidgeRBF = false;
krrSigmaGrid = [20 30 40 50 60];
krrLambdaGrid = logspace(-4, -2, 6);

% ---- Neural Net (MLP) settings ----
% Fast default: no inner tuning (set nnTune=true if you want it)
nnTune   = false;
useFitrnet = false; %exist("fitrnet","file") == 2;
% Fixed NN hyperparameters (used when nnTune=false)
nnHidden = [128 64];     % hidden layer sizes
nnLambda = 1e-3;         % fitrnet: L2 regularization strength (Lambda)
nnEpochs = 200;
% Optional tuning grid (only used when nnTune=true)
nnHiddenGrid = { [64], [128], [128 64] };
nnLambdaGrid = [1e-4, 1e-3];

nnVerbose = false;       % Doesn't work. Set true to see NN training output
% =======================

%% Load tables
tic
Tlab = readtable(labelCsv);
Temb = readtable(embCsv);

% ---- filter to selected raters ----
Tlab = filterByRaters(Tlab, raterSelection);
fprintf("After rater filtering: %d rows\n", height(Tlab));

% If embeddings CSV has duplicates, keep one row per sentence_id
if duplicatedIDs(Temb.sentence_id)
    [~, ia] = unique(string(Temb.sentence_id), 'stable');
    Temb = Temb(ia, :);
end

% Detect embedding columns
embCols = findEmbeddingColumns(Temb);
fprintf("Detected %d embedding columns.\n", numel(embCols));

% Join labels + embeddings on sentence_id
T = innerjoin(Tlab, Temb(:, ["sentence_id", embCols]), "Keys", "sentence_id");
fprintf("Joined table rows: %d\n", height(T));

% Build X and Y
X = table2array(T(:, embCols));
Y = [T.x_m, T.y_m, T.amplitude, T.radius];
yNames = ["x_m","y_m","amplitude","radius"];

fprintf("Embedding dimensionality in X: %d\n", size(X,2));
if size(X,2) < 50
    warning("Embedding dimension seems suspiciously small (%d). Check your embedding columns.", size(X,2));
end

%% Grouped CV by sentence_id (prevents leakage)
sentIDs = string(T.sentence_id);
uIDs = unique(sentIDs);
cvOuter = cvpartition(numel(uIDs), "KFold", K_outer);

Yhat_ridge = nan(size(Y));
Yhat_pls   = nan(size(Y));
Yhat_krr   = nan(size(Y));
Yhat_nn    = nan(size(Y));     % <-- NEW: neural net predictions

chosenLambda = nan(K_outer,1);
chosenNComp  = nan(K_outer,1);

chosenKrrSigma  = nan(K_outer,1);
chosenKrrLambda = nan(K_outer,1);

chosenNnHidden = strings(K_outer,1);
chosenNnLambda = nan(K_outer,1);

% Decide NN backend
useFFN     = exist("feedforwardnet","file") == 2;
if ~useFitrnet && ~useFFN
    warning("No NN backend found (fitrnet or feedforwardnet). Neural net model will be skipped.");
end
if useFitrnet
    fprintf("Neural net backend: fitrnet (Statistics and Machine Learning Toolbox)\n");
elseif useFFN
    fprintf("Neural net backend: feedforwardnet (Deep Learning Toolbox)\n");
end

for fold = 1:K_outer
    testUIDs  = uIDs(test(cvOuter, fold));
    trainUIDs = uIDs(training(cvOuter, fold));

    isTest  = ismember(sentIDs, testUIDs);
    isTrain = ismember(sentIDs, trainUIDs);

    Xtr = X(isTrain,:);
    Ytr = Y(isTrain,:);
    Xte = X(isTest,:);

    % Standardize X based on training fold
    [XtrZ, muX, sigX] = zscore(Xtr);
    sigX(sigX==0) = 1;
    XteZ = (Xte - muX) ./ sigX;

    % Standardize Y also
    muY = mean(Ytr);
    sigY = std(Ytr);
    sigY(sigY== 0) = 1;
    YtrZ = (Ytr - muY) ./ sigY;

    trainSentIDs = string(T.sentence_id(isTrain));

    %% ---- Ridge: choose lambda via inner grouped CV ----
    chosenLambda(fold) = selectRidgeLambdaGrouped(XtrZ, YtrZ, trainSentIDs, lambdaGrid, K_inner);
    lam = chosenLambda(fold);

    B_ridge = fitRidgeMultiOutput(XtrZ, YtrZ, lam);
    Yhat_ridge(isTest,:) = muY + sigY.*predictWithIntercept(XteZ, B_ridge);

    %% ---- PLS2: choose nComp via inner grouped CV ----
    p = size(XtrZ,2);
    nCompGrid = 1:min([maxPLSComp, p, size(XtrZ,1)-2]);
    chosenNComp(fold) = selectPLSComponentsGrouped(XtrZ, YtrZ, trainSentIDs, nCompGrid, K_inner);

    nComp = chosenNComp(fold);
    [~,~,~,~,B_pls] = plsregress(XtrZ, YtrZ, nComp);
    Yhat_pls(isTest,:) = muY + sigY.*([ones(sum(isTest),1), XteZ] * B_pls);

    %% ---- Kernel Ridge (RBF): choose sigma & lambda via inner grouped CV ----
    if tuneRidgeRBF
        [bestSigma, bestLamK] = selectKRRHyperparamsGrouped_parfor(XtrZ, YtrZ, trainSentIDs, krrSigmaGrid, krrLambdaGrid, K_inner);
    else
        bestSigma = median(krrSigmaGrid);
        bestLamK = median(krrLambdaGrid);
    end

    chosenKrrSigma(fold)  = bestSigma;
    chosenKrrLambda(fold) = bestLamK;

    krrModel = fitKRR(XtrZ, YtrZ, bestSigma, bestLamK);
    Yhat_krr(isTest,:) = muY + sigY.*predictKRR(krrModel, XteZ);

    %% ---- Neural Net (MLP): fixed hyperparams or tuned ----
    if useFitrnet || useFFN
        if nnTune
            [bestHidden, bestNnLam] = selectNNHyperparamsGrouped_parfor( ...
                XtrZ, YtrZ, trainSentIDs, nnHiddenGrid, nnLambdaGrid, K_inner, nnEpochs, useFitrnet, nnVerbose);
        else
            bestHidden = nnHidden;
            bestNnLam  = nnLambda;
        end

        chosenNnHidden(fold) = hiddenToString(bestHidden);
        chosenNnLambda(fold) = bestNnLam;

        Yhat_nn(isTest,:) = muY + sigY.*fitPredictNNmulti(XtrZ, YtrZ, XteZ, bestHidden, bestNnLam, nnEpochs, useFitrnet, nnVerbose);
    end

    fprintf("Fold %d/%d: ridge=%.3g | pls=%d | KRR(s=%.3g,l=%.3g) | NN(h=%s, lam=%.2g)\n", ...
        fold, K_outer, lam, nComp, bestSigma, bestLamK, chosenNnHidden(fold), chosenNnLambda(fold));
end

%% Metrics
yt = Y(:,4);
yp = Yhat_nn(:,4);

fprintf("Radius true:   min=%.4f mean=%.4f max=%.4f\n", min(yt), mean(yt), max(yt));
fprintf("Radius predNN: min=%.4f mean=%.4f max=%.4f\n", min(yp), mean(yp), max(yp));
fprintf("Pred outside [0.05, 0.24]: %d of %d\n", sum(yp < 0.05 | yp > 0.24), numel(yp));

metrics_ridge = regressionMetrics(Y, Yhat_ridge, yNames);
metrics_pls   = regressionMetrics(Y, Yhat_pls,   yNames);
metrics_krr   = regressionMetrics(Y, Yhat_krr,   yNames);
metrics_nn    = regressionMetrics(Y, Yhat_nn,    yNames);

disp("=== Ridge (grouped CV) ==="); disp(metrics_ridge);
disp("=== PLS2 (grouped CV) ==="); disp(metrics_pls);
disp("=== Kernel Ridge RBF (grouped CV) ==="); disp(metrics_krr);
disp("=== Neural Net (MLP) (grouped CV) ==="); disp(metrics_nn);

%% Plot R^2 comparison
figure('Name','R^2 by output');
bar([metrics_ridge.R2, metrics_pls.R2, metrics_krr.R2, metrics_nn.R2]); grid on;
set(gca, 'XTickLabel', yNames);
ylabel('R^2 (out-of-fold)');
legend({'Ridge','PLS2','Kernel Ridge (RBF)','Neural Net'}, 'Location','best');
title(sprintf('Grouped %d-fold CV by sentence\\_id | raters=%s', K_outer, mat2str(raterSelection)));

%% Plot RMSE comparison
figure('Name','RMSE by output');
bar([metrics_ridge.RMSE, metrics_pls.RMSE, metrics_krr.RMSE, metrics_nn.RMSE]); grid on;
set(gca, 'XTickLabel', yNames);
ylabel('RMSE');
legend({'Ridge','PLS2','Kernel Ridge (RBF)','Neural Net'}, 'Location','best');
title('RMSE (out-of-fold)');

%% Pred vs actual plots
plotPredVsActual(Y, Yhat_ridge, yNames, "Ridge: Predicted vs Actual");
plotPredVsActual(Y, Yhat_pls,   yNames, "PLS2: Predicted vs Actual");
plotPredVsActual(Y, Yhat_krr,   yNames, "Kernel Ridge (RBF): Predicted vs Actual");
plotPredVsActual(Y, Yhat_nn,    yNames, "Neural Net: Predicted vs Actual");

%% Hyperparam summaries
fprintf("\nChosen ridge lambdas per fold:\n"); disp(chosenLambda);
fprintf("Chosen PLS nComp per fold:\n"); disp(chosenNComp);
fprintf("Chosen KRR sigma per fold:\n"); disp(chosenKrrSigma);
fprintf("Chosen KRR lambda per fold:\n"); disp(chosenKrrLambda);
fprintf("Chosen NN hidden per fold:\n"); disp(chosenNnHidden);
fprintf("Chosen NN lambda per fold:\n"); disp(chosenNnLambda);
toc

%% ====================== Local functions ======================

function TlabOut = filterByRaters(TlabIn, raterSelection)
    if ~ismember("rater_id", string(TlabIn.Properties.VariableNames))
        error("label CSV must contain a 'rater_id' column.");
    end

    rid = string(TlabIn.rater_id);

    if isnumeric(raterSelection)
        if any(~ismember(raterSelection, [1 2 3]))
            error("Numeric raterSelection must be subset of [1 2 3]. Got: %s", mat2str(raterSelection));
        end
        wanted = "rater_" + string(raterSelection(:));
    else
        wanted = string(raterSelection(:));
    end

    mask = ismember(rid, wanted);
    if ~any(mask)
        error("No rows matched the selected raters: %s", strjoin(wanted, ", "));
    end
    TlabOut = TlabIn(mask, :);

    fprintf("Using raters: %s\n", strjoin(unique(string(TlabOut.rater_id)), ", "));
end

function tf = duplicatedIDs(ids)
    ids = string(ids);
    tf = numel(unique(ids)) < numel(ids);
end

function embCols = findEmbeddingColumns(Temb)
    vnames = string(Temb.Properties.VariableNames);
    isEmb = startsWith(vnames, "emb_") | startsWith(vnames, "emb");

    isNum = false(size(vnames));
    for i=1:numel(vnames)
        isNum(i) = isnumeric(Temb.(vnames(i)));
    end

    embCols = vnames(isEmb & isNum);

    if isempty(embCols)
        error("Could not detect embedding columns. Rename them to start with 'emb_' (e.g., emb_000..emb_767).");
    end

    embCols = sortEmbeddingColsByIndex(embCols);
end

function embColsSorted = sortEmbeddingColsByIndex(embCols)
    idx = nan(numel(embCols),1);
    for i=1:numel(embCols)
        name = embCols(i);
        tok = regexp(name, "(\d+)$", "tokens", "once");
        if ~isempty(tok)
            idx(i) = str2double(tok{1});
        end
    end
    if all(isfinite(idx))
        [~, ord] = sort(idx, "ascend");
        embColsSorted = embCols(ord);
    else
        embColsSorted = embCols;
    end
end

%% ----- Ridge helpers -----
function B = fitRidgeMultiOutput(XtrZ, Ytr, lambda)
    [n,p] = size(XtrZ);
    X1 = [ones(n,1), XtrZ];
    I = eye(p+1); I(1,1) = 0;  % don't penalize intercept
    B = (X1'*X1 + lambda*I) \ (X1'*Ytr);
end

function Yhat = predictWithIntercept(XteZ, B)
    n = size(XteZ,1);
    Yhat = [ones(n,1), XteZ] * B;
end

function bestLambda = selectRidgeLambdaGrouped(XtrZ, Ytr, trainSentIDs, lambdaGrid, Kinner)
    u = unique(trainSentIDs);
    c = cvpartition(numel(u), "KFold", Kinner);
    msePerLambda = zeros(numel(lambdaGrid),1);

    for li = 1:numel(lambdaGrid)
        lam = lambdaGrid(li);
        foldMSE = zeros(Kinner,1);

        for k = 1:Kinner
            testUIDs = u(test(c,k));
            isTe = ismember(trainSentIDs, testUIDs);
            isTr = ~isTe;

            B = fitRidgeMultiOutput(XtrZ(isTr,:), Ytr(isTr,:), lam);
            Yhat = predictWithIntercept(XtrZ(isTe,:), B);
            foldMSE(k) = mean(mean((Ytr(isTe,:) - Yhat).^2, 1), 2);
        end

        msePerLambda(li) = mean(foldMSE);
    end

    [~, idx] = min(msePerLambda);
    bestLambda = lambdaGrid(idx);
end

%% ----- PLS helpers -----
function bestNComp = selectPLSComponentsGrouped(XtrZ, Ytr, trainSentIDs, nCompGrid, Kinner)
    u = unique(trainSentIDs);
    c = cvpartition(numel(u), "KFold", Kinner);
    msePerComp = zeros(numel(nCompGrid),1);

    parfor ci = 1:numel(nCompGrid)
        nc = nCompGrid(ci);
        foldMSE = zeros(Kinner,1);

        for k = 1:Kinner
            testUIDs = u(test(c,k));
            isTe = ismember(trainSentIDs, testUIDs);
            isTr = ~isTe;

            [~,~,~,~,B] = plsregress(XtrZ(isTr,:), Ytr(isTr,:), nc);
            Yhat = [ones(sum(isTe),1), XtrZ(isTe,:)] * B;
            foldMSE(k) = mean(mean((Ytr(isTe,:) - Yhat).^2, 1), 2);
        end

        msePerComp(ci) = mean(foldMSE);
    end

    [~, idx] = min(msePerComp);
    bestNComp = nCompGrid(idx);
end

%% ----- Kernel Ridge (RBF) helpers -----
function K = rbfKernel(X1, X2, sigma)
    X1sq = sum(X1.^2, 2);
    X2sq = sum(X2.^2, 2)';
    D2 = max(0, X1sq + X2sq - 2*(X1*X2'));
    K = exp(-D2 / (2*sigma^2));
end

function model = fitKRR(XtrZ, Ytr, sigma, lambda)
    n = size(XtrZ,1);
    K = rbfKernel(XtrZ, XtrZ, sigma);
    A = K + lambda * eye(n);

    R = chol(A, 'lower');
    Alpha = R' \ (R \ Ytr);

    model.XtrZ = XtrZ;
    model.sigma = sigma;
    model.lambda = lambda;
    model.Alpha = Alpha;
end

function Yhat = predictKRR(model, XteZ)
    Kte = rbfKernel(XteZ, model.XtrZ, model.sigma);
    Yhat = Kte * model.Alpha;
end

function [bestSigma, bestLambda] = selectKRRHyperparamsGrouped_parfor(XtrZ, Ytr, trainSentIDs, sigmaGrid, lambdaGrid, Kinner)
    u = unique(trainSentIDs);
    c = cvpartition(numel(u), "KFold", Kinner);

    foldMasks = cell(Kinner,1);
    for k = 1:Kinner
        testUIDs = u(test(c,k));
        isTe = ismember(trainSentIDs, testUIDs);
        foldMasks{k}.isTe = isTe;
        foldMasks{k}.isTr = ~isTe;
    end

    [S,L] = ndgrid(sigmaGrid, lambdaGrid);
    combos = [S(:), L(:)];
    nComb = size(combos,1);

    mseCombo = nan(nComb,1);

    parfor ci = 1:nComb
        sigma = combos(ci,1);
        lam   = combos(ci,2);

        foldMSE = zeros(Kinner,1);
        for k = 1:Kinner
            isTr = foldMasks{k}.isTr;
            isTe = foldMasks{k}.isTe;

            mdl = fitKRR(XtrZ(isTr,:), Ytr(isTr,:), sigma, lam);
            Yhat = predictKRR(mdl, XtrZ(isTe,:));
            foldMSE(k) = mean(mean((Ytr(isTe,:) - Yhat).^2, 1), 2);
        end

        mseCombo(ci) = mean(foldMSE);
    end

    [~, bestIdx] = min(mseCombo);
    bestSigma  = combos(bestIdx,1);
    bestLambda = combos(bestIdx,2);
end

%% ----- Neural Net (MLP) helpers -----
function s = hiddenToString(h)
    if isempty(h); s = "[]"; return; end
    s = "[" + strjoin(string(h), " ") + "]";
end

function Yhat = fitPredictNNmulti(XtrZ, Ytr, XteZ, hidden, lambda, epochs, useFitrnet, verbose)
    % Fits 4 separate NN regressors (one per output) and returns predictions (nTest x 4)
    nOut = size(Ytr,2);
    Yhat = zeros(size(XteZ,1), nOut);
    for j = 1:nOut
        Yhat(:,j) = fitPredictNNsingle(XtrZ, Ytr(:,j), XteZ, hidden, lambda, epochs, useFitrnet, verbose);
    end
end

function yhat = fitPredictNNsingle(XtrZ, ytr, XteZ, hidden, lambda, epochs, useFitrnet, verbose)
    if useFitrnet
        % fitrnet supports single-output regression
        args = {'LayerSizes', hidden, 'Lambda', lambda, 'Standardize', false};
        try
            mdl = fitrnet(XtrZ, ytr, args{:}, 'IterationLimit', epochs);
        catch
            % older versions may not accept IterationLimit
            mdl = fitrnet(XtrZ, ytr, args{:});
        end
        yhat = predict(mdl, XteZ);

    else
        % feedforwardnet fallback (Deep Learning Toolbox)
        if exist("feedforwardnet","file") ~= 2
            error("No neural net function available (fitrnet or feedforwardnet).");
        end

        net = feedforwardnet(hidden, 'trainscg');

        net.inputs{1}.processFcns  = {};     % disable removeconstantrows/mapminmax
        net.outputs{end}.processFcns = {};   % disable output mapminmax

        net.trainParam.epochs = epochs;
        net.trainParam.showWindow = false;
        net.trainParam.showCommandLine = verbose;

        % We already handle splitting via CV; train on all given data
        net.divideFcn = 'dividetrain';

        % Regularization parameter is 0..1 in feedforwardnet; map lambda if tiny
        reg = lambda;
        if reg < 0 || reg > 1
            % if user passes something like 1e-3, keep it but clip to [0,1]
            reg = max(0, min(1, reg));
        end
        net.performParam.regularization = reg;

        % Train (note transpose: features x N)
        net = train(net, XtrZ', ytr');
        yhat = net(XteZ')';
    end
end

function [bestHidden, bestLambda] = selectNNHyperparamsGrouped_parfor(XtrZ, Ytr, trainSentIDs, hiddenGrid, lambdaGrid, Kinner, epochs, useFitrnet, verbose)
    % Light inner-CV tuning for NN. Can still be expensive; keep grids small.

    u = unique(trainSentIDs);
    c = cvpartition(numel(u), "KFold", Kinner);

    foldMasks = cell(Kinner,1);
    for k = 1:Kinner
        testUIDs = u(test(c,k));
        isTe = ismember(trainSentIDs, testUIDs);
        foldMasks{k}.isTe = isTe;
        foldMasks{k}.isTr = ~isTe;
    end

    nH = numel(hiddenGrid);
    nL = numel(lambdaGrid);
    nComb = nH * nL;

    mseCombo = nan(nComb,1);
    combHiddenIdx = nan(nComb,1);
    combLam = nan(nComb,1);

    idx = 0;
    for hi = 1:nH
        for li = 1:nL
            idx = idx + 1;
            combHiddenIdx(idx) = hi;
            combLam(idx) = lambdaGrid(li);
        end
    end

    parfor ci = 1:nComb
        hidden = hiddenGrid{combHiddenIdx(ci)};
        lam = combLam(ci);

        foldMSE = zeros(Kinner,1);
        for k = 1:Kinner
            isTr = foldMasks{k}.isTr;
            isTe = foldMasks{k}.isTe;

            Yhat = fitPredictNNmulti(XtrZ(isTr,:), Ytr(isTr,:), XtrZ(isTe,:), hidden, lam, epochs, useFitrnet, verbose);
            foldMSE(k) = mean(mean((Ytr(isTe,:) - Yhat).^2, 1), 2);
        end

        mseCombo(ci) = mean(foldMSE);
    end

    [~, bestIdx] = min(mseCombo);
    bestHidden = hiddenGrid{combHiddenIdx(bestIdx)};
    bestLambda = combLam(bestIdx);
end

%% ----- Metrics + plotting -----
function M = regressionMetrics(Ytrue, Ypred, yNames)
    nOut = size(Ytrue,2);
    R2 = zeros(nOut,1);
    RMSE = zeros(nOut,1);
    MAE = zeros(nOut,1);

    for j = 1:nOut
        yt = Ytrue(:,j);
        yp = Ypred(:,j);
        ok = isfinite(yt) & isfinite(yp);
        yt = yt(ok); yp = yp(ok);

        sse = sum((yt - yp).^2);
        sst = sum((yt - mean(yt)).^2);
        R2(j) = 1 - sse/sst;

        RMSE(j) = sqrt(mean((yt - yp).^2));
        MAE(j)  = mean(abs(yt - yp));
    end

    M = table(yNames(:), R2, RMSE, MAE, 'VariableNames', ["Output","R2","RMSE","MAE"]);
end

function plotPredVsActual(Ytrue, Ypred, yNames, figTitle)
    figure('Name', figTitle);
    for j = 1:size(Ytrue,2)
        subplot(2,2,j);
        yt = Ytrue(:,j); yp = Ypred(:,j);
        ok = isfinite(yt) & isfinite(yp);
        yt = yt(ok); yp = yp(ok);

        scatter(yt, yp, 8, 'filled'); grid on; hold on;
        mn = min([yt; yp]); mx = max([yt; yp]);
        plot([mn mx], [mn mx], 'k-', 'LineWidth', 1);
        xlabel("Actual"); ylabel("Predicted");
        title(yNames(j));
    end
    sgtitle(figTitle);
end
