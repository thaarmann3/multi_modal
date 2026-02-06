% Exports final Ridge + NN with the SAME preprocessing as your CV script:
%   Xz = (X - muX)./sigX
%   Yz = (Y - muY)./sigY
% Models are trained on (Xz -> Yz). Python will unstandardize back.

clear; clc; close all; rng(1);

%% ===== USER EDITS =====
labelCsv = "sbert_potentialfield_commands_2000_sentences_3_raters_learnable.csv";
embCsv   = "embeddings_384.csv";  % or embeddings_768.csv
raterSelection = [1 2 3];

% Final hyperparameters to deploy (set from your CV outcomes)
ridgeLambda_final = 10;     % <-- set to median(chosenLambda) from CV

% Neural net settings to deploy (match your script)
nnHidden_final = [128 64];
nnLambda_final = 1e-3;        % for fitrnet (if used); for feedforwardnet this is treated as regularization (0..1)
nnEpochs_final = 200;

% Force exportable NN backend? (True => feedforwardnet so Python can match exactly)
forceFeedforwardExport = true;
% =======================

%% Load tables
Tlab = readtable(labelCsv);
Temb = readtable(embCsv);

Tlab = filterByRaters(Tlab, raterSelection);

% Deduplicate embeddings to one row per sentence_id
if duplicatedIDs(Temb.sentence_id)
    [~, ia] = unique(string(Temb.sentence_id), 'stable');
    Temb = Temb(ia,:);
end

embCols = findEmbeddingColumns(Temb);

T = innerjoin(Tlab, Temb(:, ["sentence_id", embCols]), "Keys", "sentence_id");
X = table2array(T(:, embCols));
Y = [T.x_m, T.y_m, T.amplitude, T.radius];
yNames = ["x_m","y_m","amplitude","radius"];

fprintf("Training rows: %d | Embedding dim: %d\n", size(X,1), size(X,2));

%% Standardize X and Y EXACTLY like your CV script
muX = mean(X, 1);
sigX = std(X, 0, 1); sigX(sigX==0) = 1;
Xz = (X - muX) ./ sigX;

muY = mean(Y, 1);
sigY = std(Y, 0, 1); sigY(sigY==0) = 1;
Yz = (Y - muY) ./ sigY;

%% ---- Ridge (multi-output): train Xz -> Yz ----
B_ridge = fitRidgeMultiOutput(Xz, Yz, ridgeLambda_final);  % (p+1) x 4

%% ---- Neural net: train 4 separate regressors, each predicts y_z ----
useFitrnet = exist("fitrnet","file") == 2;
useFFN     = exist("feedforwardnet","file") == 2;

if forceFeedforwardExport
    if ~useFFN
        error("forceFeedforwardExport=true but feedforwardnet not available.");
    end
    nnBackend = "feedforwardnet";
else
    if useFitrnet
        nnBackend = "fitrnet";
    elseif useFFN
        nnBackend = "feedforwardnet";
    else
        error("No NN backend available (fitrnet or feedforwardnet).");
    end
end
fprintf("NN backend for export: %s\n", nnBackend);

% Store NN parameters in exportable form:
nets = cell(1,4);

if nnBackend == "fitrnet"
    % Save the fitrnet models (MATLAB only). Python exact export not supported.
    for j = 1:4
        args = {'LayerSizes', nnHidden_final, 'Lambda', nnLambda_final, 'Standardize', false};
        try
            nets{j} = fitrnet(Xz, Yz(:,j), args{:}, 'IterationLimit', nnEpochs_final);
        catch
            nets{j} = fitrnet(Xz, Yz(:,j), args{:});
        end
    end

    % ---- Convert string arrays to SciPy-friendly types ----
    yNames_cell = cellstr(yNames);      % cell array of char
    tf1_cell = cellstr(tf1);
    tf2_cell = cellstr(tf2);
    tf3_cell = cellstr(tf3);
    nnBackend_char = char(nnBackend);   % plain char
    
    % ---- Save ONLY numeric arrays + cellstr/char (SciPy can load this) ----
    save("exported_models_like_cv.mat", ...
        "muX","sigX","muY","sigY", ...
        "B_ridge","ridgeLambda_final", ...
        "W1","b1","W2","b2","W3","b3", ...
        "tf1_cell","tf2_cell","tf3_cell", ...
        "yNames_cell","nnBackend_char", ...
        "nnHidden_final","nnLambda_final","nnEpochs_final", ...
        "-v7");

    fprintf("Saved exported_models_like_cv.mat (contains fitrnet objects; MATLAB-only)\n");

else
    % feedforwardnet: export weights/biases so Python can reproduce exactly
    H1 = nnHidden_final(1); H2 = nnHidden_final(2);
    p = size(Xz,2);

    W1 = zeros(H1, p, 4);  b1 = zeros(H1, 4);
    W2 = zeros(H2, H1,4);  b2 = zeros(H2, 4);
    W3 = zeros(1,  H2,4);  b3 = zeros(1,  4);

    tf1 = strings(1,4); tf2 = strings(1,4); tf3 = strings(1,4);

    for j = 1:4
        net = feedforwardnet(nnHidden_final, 'trainscg');
        
        net.inputs{1}.processFcns  = {};     % disable removeconstantrows/mapminmax
        net.outputs{end}.processFcns = {};   % disable output mapminmax

        net.divideFcn = 'dividetrain';
        net.trainParam.epochs = nnEpochs_final;
        net.trainParam.showWindow = false;
        net.trainParam.showCommandLine = false;

        % Your CV code "clips" lambda to [0,1] for feedforwardnet regularization.
        reg = max(0, min(1, nnLambda_final));
        net.performParam.regularization = reg;

        net = train(net, Xz', Yz(:,j)');  % NOTE: predicts y_z

        nets{j} = net;

        W1(:,:,j) = net.IW{1,1}; b1(:,j) = net.b{1};
        W2(:,:,j) = net.LW{2,1}; b2(:,j) = net.b{2};
        W3(:,:,j) = net.LW{3,2}; b3(:,j) = net.b{3};

        tf1(j) = string(net.layers{1}.transferFcn);
        tf2(j) = string(net.layers{2}.transferFcn);
        tf3(j) = string(net.layers{3}.transferFcn);
    end

    nnReg_final = max(0, min(1, nnLambda_final));

    % ---- Convert string arrays to SciPy-friendly types ----
    yNames_cell = cellstr(yNames);      % cell array of char
    tf1_cell = cellstr(tf1);
    tf2_cell = cellstr(tf2);
    tf3_cell = cellstr(tf3);
    nnBackend_char = char(nnBackend);   % plain char
    
    % ---- Save ONLY numeric arrays + cellstr/char (SciPy can load this) ----
    save("exported_models_like_cv.mat", ...
        "muX","sigX","muY","sigY", ...
        "B_ridge","ridgeLambda_final", ...
        "W1","b1","W2","b2","W3","b3", ...
        "tf1_cell","tf2_cell","tf3_cell", ...
        "yNames_cell","nnBackend_char", ...
        "nnHidden_final","nnLambda_final","nnEpochs_final", ...
        "-v7");

    fprintf("Saved exported_models_like_cv.mat (exportable weights/biases)\n");
end

%% ---------- helpers ----------
function TlabOut = filterByRaters(TlabIn, raterSelection)
    rid = string(TlabIn.rater_id);
    wanted = "rater_" + string(raterSelection(:));
    mask = ismember(rid, wanted);
    if ~any(mask), error("No rows matched selected raters."); end
    TlabOut = TlabIn(mask,:);
end

function tf = duplicatedIDs(ids)
    ids = string(ids);
    tf = numel(unique(ids)) < numel(ids);
end

function embCols = findEmbeddingColumns(Temb)
    vnames = string(Temb.Properties.VariableNames);
    isEmb = startsWith(vnames, "emb_") | startsWith(vnames, "emb");
    isNum = false(size(vnames));
    for i=1:numel(vnames), isNum(i) = isnumeric(Temb.(vnames(i))); end
    embCols = vnames(isEmb & isNum);
    if isempty(embCols)
        error("No embedding columns found. Expect names like emb_000..");
    end
end

function B = fitRidgeMultiOutput(XtrZ, YtrZ, lambda)
    [n,p] = size(XtrZ);
    X1 = [ones(n,1), XtrZ];
    I = eye(p+1); I(1,1) = 0;
    B = (X1'*X1 + lambda*I) \ (X1'*YtrZ);
end