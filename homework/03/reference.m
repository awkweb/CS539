rng(1);
% ==========================LOADING DATA===================================
filename = 'data/adult/adult.dat'; adult_data = readtable(filename);
adult_data.Properties.VariableNames = {'age', 'work_class', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country','salary'};

load data/optdigits/optdigits_test.dat; load data/optdigits/optdigits_train.dat;
% ==========================MODEL CONSTRUCTION=============================
cvp_indices = cvpartition(height(adult_data), 'holdout', 0.25);
adult_data_cvp = adult_data(training(cvp_indices),:);
X_ctree = adult_data_cvp(:,{'age', 'work_class', 'education_num', 'marital_status', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week'});
Y_ctree = adult_data_cvp(:,{'salary'});
ctree = fitctree(X_ctree, Y_ctree, 'MaxNumSplits', 20, 'MinLeafSize', 8, 'PredictorSelection', 'curvature', 'Surrogate', 'on');

X_rtree = adult_data_cvp(:,{'age', 'work_class', 'salary', 'marital_status', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week'});
Y_rtree = adult_data_cvp(:,{'education_num'});
rtree = fitrtree(X_rtree, Y_rtree, 'MaxNumSplits', 20, 'MinLeafSize', 8, 'PredictorSelection', 'curvature', 'Surrogate', 'on');
% rtree = fitrtree(X_rtree, 'education_num~age+salary+sex');

X_csvm = adult_data_cvp(1:30,{'age', 'work_class', 'education_num', 'marital_status', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week'});
Y_csvm = adult_data_cvp(1:30,{'salary'});
csvm_linear = fitcsvm(X_csvm, Y_csvm, 'KernelFunction', 'linear'); %'rbf'
csvm_polynomial = fitcsvm(X_csvm, Y_csvm, 'KernelFunction', 'polynomial', 'PolynomialOrder', 4);
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
% net.trainParam.epochs = 5000; net.trainParam.goal = 0; net.trainParam.lr = 0.01; net.trainParam.max_fail = 6; net.trainParam.min_grad = 1e-5; net.trainParam.time = inf;
% ==========================CROSS VALIDATION===============================
disp('======================CLASSIFICATION TREE==========================')
cv_ctree = crossval(ctree, 'Kfold', 4);
ctree_time_start = tic;
ctree_label = kfoldPredict(cv_ctree);
ctree_time_elapsed = toc(ctree_time_start);
cv_ctree_error = kfoldLoss(cv_ctree); cv_ctree_accuracy = 1 - cv_ctree_error;

index = randsample(numel(ctree_label),10);
table(Y_ctree(index,:).salary, ctree_label(index,:), 'VariableNames', {'TrueLabels','PredictedLabels'})
% view(cv_tree.Trained{1}, 'Mode', 'graph');
cv_ctree_confusionmat = confusionmat(Y_ctree.salary, ctree_label);
disp('======================REGRESSION TREE==============================')
cv_rtree = crossval(rtree, 'Kfold', 4);
rtree_label = kfoldPredict(cv_rtree);
sse = mean(kfoldfun(cv_rtree, @sseEval)); rmse = mean(kfoldfun(cv_rtree, @rmseEval)); rse = mean(kfoldfun(cv_rtree, @rseEval)); r2 = mean(kfoldfun(cv_rtree, @r2Eval));
disp('======================SUPPORT VECTOR MACHINES======================')
cv_csvm = crossval(csvm_linear, 'Kfold', 4);
csvm_label = kfoldPredict(cv_csvm);
cv_csvm_error = kfoldLoss(cv_csvm); cv_csvm_accuracy = 1 - cv_csvm_error;
% csvm_sv = csvm.SupportVectors;
disp('======================NEURAL NETWORK===============================')
net = train(net, X_net_train, Y_net_train);
y = net(X_net_train); perf = perform(net, Y_net_train, y);
Y_score = net(X_net_test); Y_pred = round(Y_score);

[rocx, rocy, roct, auc] = perfcurve(Y_net_test, Y_score, 1);
figure; plot(rocx, rocy);
title('Neural Network ROC'); grid on;
xlabel('False positive [FP/(TN+FP)]'); ylabel('True positive [TP/(TP+FN)]');

index = randsample(numel(Y_pred),10);
table(Y_net_test(:,index)', Y_pred(:,index)', 'VariableNames', {'TrueLabels','PredictedLabels'})
% ==========================FUNCTIONS======================================
function G = mysigmoid(U, V)
    gamma = 0.5; c = -1; G = tanh(gamma*U'*V + c);
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
    rmse = sqrt(mean((Ytest-mean(Ytrain)).^2)); rmse0 = std(Ytrain-mean(Ytrain)); r2 = 1 - (rmse/rmse0);
end
% precision_train = mean(precision(confusion_train));
% recall_train = mean(recall(confusion_train));
% function y = precision(M) y = diag(M) ./ sum(M,2); end
% function y = recall(M) y = diag(M) ./ sum(M,1)'; end
function [ c, performance, net ] = NeuralNet(trainingData, testData)
    rng(1) % For reproducibility
    trainX = trainingData(1:64,:); trainY=ind2vec(trainingData(65,:)+1,10);
    testX = testData(1:64, :); testY = ind2vec(testData(65, :) + 1, 10);

    % Create a Pattern Recognition Network
    hiddenLayerSize = [200, 40]; net = patternnet(hiddenLayerSize);
    % net.trainFcn = 'traingd'; % 'trainscg' is default

    %net.trainParam.epochs = 400;
    %net.trainParam.lr = 0.01;
    % net.performParam.regularization = 1e-5;
    % net.performParam.normalization;

    % Train
    [net,tr] = train(net, trainX, trainY);

    % Test
    outputs = net(testX); errors = gsubtract(testY, outputs);
    performance = perform(net, testY, outputs);
    [c,cm,ind,per] = confusion(testY, outputs);
    % c=% misclassified, cm=matrix

    % view(net)

    %figure, plotperform(tr) %figure, plottrainstate(tr)
    figure, plotconfusion(testY,outputs) %figure, ploterrhist(errors)
end