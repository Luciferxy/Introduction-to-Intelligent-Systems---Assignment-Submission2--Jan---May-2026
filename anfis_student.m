%% ============================================================
%  HYBRID NEURO-FUZZY STUDENT PERFORMANCE SYSTEM  (MATLAB)
%  Author  : Sourav | Course: Soft Computing, MUJ
%  Method  : ANFIS (Adaptive Neuro-Fuzzy Inference System)
%            via MATLAB Fuzzy Logic + Neural Network Toolbox
%% ============================================================
clc; clear; close all;

%% ── 1. GENERATE SYNTHETIC DATASET ───────────────────────────
rng(42);
N = 500;

% Generate 3 clusters: Poor (0), Average (1), Good (2)
labels_raw = randsample([0,1,2], N, true, [0.30, 0.40, 0.30]);

att = zeros(N,1); asn = zeros(N,1); tst = zeros(N,1);
for i = 1:N
    switch labels_raw(i)
        case 0   % Poor
            att(i) = clip(normrnd(50, 10), 30, 65);
            asn(i) = clip(normrnd(38, 10), 20, 55);
            tst(i) = clip(normrnd(38, 10), 20, 55);
        case 1   % Average
            att(i) = clip(normrnd(70, 8),  55, 85);
            asn(i) = clip(normrnd(58, 8),  45, 72);
            tst(i) = clip(normrnd(58, 8),  45, 72);
        case 2   % Good
            att(i) = clip(normrnd(88, 6),  75, 100);
            asn(i) = clip(normrnd(82, 8),  68, 100);
            tst(i) = clip(normrnd(82, 8),  68, 100);
    end
end

X = [att, asn, tst];
y = labels_raw' + 1;   % ANFIS needs output ≥ 1 → 1=Poor, 2=Avg, 3=Good

%% ── 2. TRAIN-TEST SPLIT ─────────────────────────────────────
splitIdx = round(0.8 * N);
Xtr = X(1:splitIdx, :);    ytr = y(1:splitIdx);
Xte = X(splitIdx+1:end,:); yte = y(splitIdx+1:end);

%% ── 3. GENERATE INITIAL FIS  (ANFIS uses Sugeno-type) ───────
%  genfis1 partitions each input into nMFs Gaussian MFs
opt     = genfisOptions('GridPartition');
opt.NumMembershipFunctions = 3;    % Low / Medium / High
opt.InputMembershipFunctionType = 'gaussmf';

initFIS = genfis([Xtr, ytr], opt);  % auto-generates Sugeno FIS

%% ── 4. ANFIS TRAINING (Hybrid: LSQ + Backprop) ──────────────
anfisOpt = anfisOptions('InitialFIS', initFIS, ...
                        'EpochNumber', 150, ...
                        'InitialStepSize', 0.01, ...
                        'DisplayANFISInformation', false, ...
                        'DisplayErrorValues', true, ...
                        'DisplayStepSize', false, ...
                        'DisplayFinalResults', true);

[trainedFIS, trainError, ~, chkFIS, chkError] = ...
    anfis([Xtr, ytr], anfisOpt);

fprintf('\nANFIS training complete.\n');
fprintf('Final training RMSE : %.4f\n', trainError(end));

%% ── 5. EVALUATE ─────────────────────────────────────────────
yPred_raw = evalfis(trainedFIS, Xte);
yPred     = round(yPred_raw);             % snap to 1 / 2 / 3
yPred     = max(1, min(3, yPred));        % clamp

acc = mean(yPred == yte);
fprintf('Test Accuracy : %.2f%%\n', acc * 100);

%% ── 6. DISPLAY MEMBERSHIP FUNCTIONS (after learning) ────────
figure('Name','Learned Membership Functions','Color','w');
featureNames = {'Attendance','Assignment Marks','Test Marks'};
for fi = 1:3
    subplot(1,3,fi);
    plotmf(trainedFIS, 'input', fi);
    title(featureNames{fi}, 'FontWeight','bold');
    xlabel('Score (0-100)'); ylabel('Membership');
end
sgtitle('Learned MFs after ANFIS Training','FontSize',13,'FontWeight','bold');

%% ── 7. TRAINING ERROR CURVE ─────────────────────────────────
figure('Name','Training Error','Color','w');
plot(trainError, 'b-', 'LineWidth', 2);
xlabel('Epoch'); ylabel('RMSE');
title('ANFIS Training Error Curve', 'FontWeight','bold');
grid on;

%% ── 8. CONFUSION MATRIX ─────────────────────────────────────
figure('Name','Confusion Matrix','Color','w');
cm = confusionchart(yte, yPred, ...
    'ColumnSummary','column-normalized', ...
    'RowSummary','row-normalized', ...
    'Title','Performance Classification');
cm.ClassLabels = {'Poor','Average','Good'};

%% ── 9. DEMO PREDICTION ──────────────────────────────────────
demoInput = [72, 68, 74];
rawOut     = evalfis(trainedFIS, demoInput);
predClass  = round(max(1, min(3, rawOut)));
classNames = {'Poor', 'Average', 'Good'};
fprintf('\nDemo: att=%d  asn=%d  tst=%d\n', demoInput);
fprintf('Raw ANFIS output : %.3f\n', rawOut);
fprintf('Predicted class  : %s\n', classNames{predClass});

%% ── Helper ───────────────────────────────────────────────────
function v = clip(x, lo, hi)
    v = max(lo, min(hi, x));
end
