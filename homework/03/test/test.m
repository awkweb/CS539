% =========================================================================
% CODE SNIPPETS FOR TEST 3
% =========================================================================
rng(1);

% =========================================================================
% LOADING DATA
% =========================================================================
filename = 'data/adult/adult.dat';
adult_data = readtable(filename);
adult_data.Properties.VariableNames = {'age', 'work_class', 'fnlwgt',...
    'education', 'education_num', 'marital_status', 'occupation',...
    'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',...
    'hours_per_week', 'native_country','salary'};

filename = 'data/adult/adult.test.dat';
adult_data_test = readtable(filename);
adult_data_test.Properties.VariableNames = {'age', 'work_class', 'fnlwgt',...
    'education', 'education_num', 'marital_status', 'occupation',...
    'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',...
    'hours_per_week', 'native_country','salary'};

load data/optdigits/optdigits_test.dat
load data/optdigits/optdigits_train.dat

% =========================================================================
% DATA PREPROCESSING
% =========================================================================

% Use Excel/Python to remove nulls, add dummy variables, etc.

% =========================================================================
% MODEL CONSTRUCTION
% =========================================================================
cvp_indices = cvpartition(height(adult_data), 'holdout', 0.25);
adult_data_cvp = adult_data(training(cvp_indices),:);
X_ctree = adult_data_cvp(:,{'age', 'work_class', 'education_num', 'marital_status',...
                  'race', 'sex', 'capital_gain', 'capital_loss',...
                  'hours_per_week'});
Y_ctree = adult_data_cvp(:,{'salary'});
ctree = fitctree(X_ctree, Y_ctree, 'MaxNumSplits', 20, 'MinLeafSize', 8,...
                'PredictorSelection', 'curvature', 'Surrogate', 'on');

X_rtree = adult_data_cvp(:,{'age', 'work_class', 'salary', 'marital_status',...
                  'race', 'sex', 'capital_gain', 'capital_loss',...
                  'hours_per_week'});
Y_rtree = adult_data_cvp(:,{'education_num'});
rtree = fitrtree(X_rtree, Y_rtree, 'MaxNumSplits', 20, 'MinLeafSize', 8,...
                 'PredictorSelection', 'curvature', 'Surrogate', 'on');
% rtree = fitrtree(X_rtree, 'education_num~age+salary+sex');

X_csvm = adult_data_cvp(1:30,{'age', 'work_class', 'education_num', 'marital_status',...
                  'race', 'sex', 'capital_gain', 'capital_loss',...
                  'hours_per_week'});
Y_csvm = adult_data_cvp(1:30,{'salary'});
csvm_linear = fitcsvm(X_csvm, Y_csvm, 'KernelFunction', 'linear');
csvm_polynomial = fitcsvm(X_csvm, Y_csvm, 'KernelFunction', 'polynomial',...
                          'PolynomialOrder', 4);
csvm_rbf = fitcsvm(X_csvm, Y_csvm, 'KernelFunction', 'rbf');
% csvm_sig = fitcsvm(X_csvm, Y_csvm, 'KernelFunction', mysigmoid);

X_net = optdigits_train(:,1:end-1);
Y_net = optdigits_train(:,end);
[m,n] = size(optdigits_train);
cvp_indices = cvpartition(m, 'holdout', 0.25);
X_net_train = X_net(training(cvp_indices),:)';
Y_net_train = Y_net(training(cvp_indices),:)';
X_net_test = X_net(test(cvp_indices),:)';
Y_net_test = Y_net(test(cvp_indices),:)';
net = patternnet(2, 'traingd');
% net.trainParam.epochs = 5000;
% net.trainParam.goal = 0;
% net.trainParam.lr = 0.01;
% net.trainParam.max_fail = 6;
% net.trainParam.min_grad = 1e-5;
% net.trainParam.time = inf;

% =========================================================================
% CROSS VALIDATION
% =========================================================================
disp('CLASSIFICATION TREE');
disp('===================================================================')
cv_ctree = crossval(ctree, 'Kfold', 4);
ctree_time_start = tic;
ctree_label = kfoldPredict(cv_ctree);
ctree_time_elapsed = toc(ctree_time_start);
cv_ctree_error = kfoldLoss(cv_ctree);
cv_ctree_accuracy = 1 - cv_ctree_error;

index = randsample(numel(ctree_label),10);
table(Y_ctree(index,:).salary, ctree_label(index,:),...
    'VariableNames', {'TrueLabels','PredictedLabels'})

% view(cv_tree.Trained{1}, 'Mode', 'graph');
display(ctree_time_elapsed);
display(cv_ctree_accuracy);

cv_ctree_confusionmat = confusionmat(Y_ctree.salary, ctree_label);
display(cv_ctree_confusionmat);

disp('REGRESSION TREE');
disp('===================================================================')
cv_rtree = crossval(rtree, 'Kfold', 4);
time_start = tic;
rtree_label = kfoldPredict(cv_rtree);
time_elapsed = toc(time_start);

index = randsample(numel(rtree_label),10);
table(Y_rtree(index,:).education_num, rtree_label(index,:),...
    'VariableNames', {'TrueLabels','PredictedLabels'})

sse = mean(kfoldfun(cv_rtree, @sseEval));
rmse = mean(kfoldfun(cv_rtree, @rmseEval));
rse = mean(kfoldfun(cv_rtree, @rseEval));
r2 = mean(kfoldfun(cv_rtree, @r2Eval));

display(time_elapsed);
display(sse);
display(rmse);
display(rse);
display(r2);

disp('SVM');
disp('===================================================================')
cv_csvm = crossval(csvm_linear, 'Kfold', 4);
time_start = tic;
csvm_label = kfoldPredict(cv_csvm);
time_elapsed = toc(time_start);
cv_csvm_error = kfoldLoss(cv_csvm);
cv_csvm_accuracy = 1 - cv_csvm_error;

index = randsample(numel(csvm_label),10);
table(Y_csvm(index,:).salary, csvm_label(index,:),...
    'VariableNames', {'TrueLabels','PredictedLabels'})

display(time_elapsed);
display(cv_csvm_accuracy);

cv_csvm_confusionmat = confusionmat(Y_csvm.salary, csvm_label);
display(cv_csvm_confusionmat);

% csvm_sv = csvm.SupportVectors;

disp('NEURAL NETWORK');
disp('===================================================================')
net = train(net, X_net_train, Y_net_train);
y = net(X_net_train);
perf = perform(net, Y_net_train, y);
Y_score = net(X_net_test);
time_start = tic;
Y_pred = round(Y_score);
time_elapsed = toc(time_start);

display(time_elapsed);

[rocx, rocy, roct, auc] = perfcurve(Y_net_test, Y_score, 1);
figure;
plot(rocx, rocy)
title('Neural Network ROC');
grid on;
xlabel('False positive rate [ = FP/(TN+FP)]');
ylabel('True positive rate [ = TP/(TP+FN)]');

index = randsample(numel(Y_pred),10);
table(Y_net_test(:,index)', Y_pred(:,index)',...
    'VariableNames', {'TrueLabels','PredictedLabels'})

% percent_correct = mean(sum(Y_pred == X_net_test)/length(X_net_test) * 100);
% display(percent_correct);

% cv_net_confusionmat = confusionmat(Y_net_test, Y_pred);
% display(cv_net_confusionmat);

% =========================================================================
% FUNCTIONS (that matlab should have built-in, but doesn't :(
% =========================================================================
function G = mysigmoid(U, V)
    gamma = 0.5;
    c = -1;
    G = tanh(gamma*U'*V + c);
end

function sse = sseEval(cmp, Xtrain, Ytrain, Wtrain, Xtest, Ytest, Wtest)
    sse = mean((Ytest-mean(Ytrain)).^2);
end

function rmse = rmseEval(cmp, Xtrain, Ytrain, Wtrain, Xtest, Ytest, Wtest)
    rmse = sqrt(mean((Ytest-mean(Ytrain)).^2));
end

function rse = rseEval(cmp, Xtrain, Ytrain, Wtrain, Xtest, Ytest, Wtest)
    rse = mean((Ytest-mean(Ytrain)).^2) ./ mean((Ytest-mean(Ytrain)).^2);
end

function r2 = r2Eval(cmp, Xtrain, Ytrain, Wtrain, Xtest, Ytest, Wtest)
    rmse = sqrt(mean((Ytest-mean(Ytrain)).^2));
    rmse0 = std(Ytrain-mean(Ytrain));
    r2 = 1 - (rmse/rmse0);
end

% =========================================================================
% MISC
% =========================================================================
% Functions to generate precision and recall of tree
% precision_train = mean(precision(confusion_train));
% recall_train = mean(recall(confusion_train));
% 
% function y = precision(M)
%   y = diag(M) ./ sum(M,2);
% end
% 
% function y = recall(M)
%   y = diag(M) ./ sum(M,1)';
% end
