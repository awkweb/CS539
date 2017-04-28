% Tom Meagher

filename = 'wdbc.csv';
wdbc_data = readtable(filename);
load wdbc.data

% Statified Random Sampling
cvp_indices = cvpartition(height(wdbc_data), 'holdout', 0.25);
X = wdbc_data(:,2:end);
Y = wdbc_data(:,1);
X_train = X(training(cvp_indices),:);
Y_train = Y(training(cvp_indices),:);
X_test = X(test(cvp_indices),:);
Y_test = Y(test(cvp_indices),:);

X_net = wdbc(:,2:end);
Y_net = wdbc(:,1);
X_net_train = X_net(training(cvp_indices),:);
Y_net_train = Y_net(training(cvp_indices),:);
X_net_test = X_net(test(cvp_indices),:);
Y_net_test = Y_net(test(cvp_indices),:);


% Classification Experiments
% i k-NN
% knn = fitcknn(X_train, Y_train, 'K', 5);
% knn_label = predict(knn);
% 
% knn2 = fitcknn(X_train, Y_train, 'K', 10);
% knn2_label = predict(knn2);

% ii Decision Trees
tree = fitctree(X_train, Y_train);
tree_cv = crossval(tree, 'Kfold', 4);
start_time = tic;
tree_label = kfoldPredict(tree_cv);
tree_time_elapsed = toc(start_time);

tree2 = fitctree(X_train, Y_train, 'MaxNumSplits', 3, 'MinLeafSize', 3,...
    'PredictorSelection', 'curvature', 'Surrogate', 'on');
tree2_cv = crossval(tree2, 'Kfold', 4);
tree2_label = kfoldPredict(tree2_cv);
% view(tree_cv.Trained{1}, 'Mode', 'graph');

tree_confusion = confusionmat(Y_train.Var1, tree_label);
tree_precision = mean(precision(tree_confusion));
tree_recall = mean(recall(tree_confusion));

tree2_confusion = confusionmat(Y_train.Var1, tree2_label);
tree2_precision = mean(precision(tree2_confusion));
tree2_recall = mean(recall(tree2_confusion));

display(tree_precision);
display(tree_recall);

display(tree2_precision);
display(tree2_recall);

% iii Artificial Neural Networks
hidden_layer_size = [200, 40];
net = patternnet(hidden_layer_size);
[net,tr] = train(net, X_net_train', Y_net_train');
outputs = net(X_net_test');
errors = gsubtract(Y_net_test', outputs);
mean(errors)

hidden_layer_size2 = [100, 50, 10];
net2 = patternnet(hidden_layer_size2);
[net2,tr] = train(net2, X_net_train', Y_net_train');
outputs2 = net2(X_net_test');
errors2 = gsubtract(Y_net_test', outputs2);
mean(errors2)

% iv Support Vector Machines
% csvm = fitcsvm(X_train, Y_train);
% csvm_linear = fitcsvm(X_train, Y_train, 'KernelFunction', 'polynomial',...
%     'PolynomialOrder', 2);
% 
% csvm_cv = crossval(csvm, 'Kfold', 4);
% csvm_poly_cv = crossval(csvm_linear, 'Kfold', 4);
% 
% csvm_cv_label = kfoldPredict(csvm_cv);
% csvm_linear_cv_label = kfoldPredict(csvm_poly_cv);

% Regression Experiments
tree = fitrtree(X_train, Y_train);
tree_cv = crossval(tree, 'Kfold', 4);
start_time = tic;
tree_label = kfoldPredict(tree_cv);
tree_time_elapsed = toc(start_time);

tree2 = fitrtree(X_train, Y_train, 'Prune', 'on');
tree2_cv = crossval(tree, 'Kfold', 4);
tree2_label = kfoldPredict(tree2_cv);

% Experiments with PCA Preprocessing
% [COEFF, SCORE] = pca(wdbc_data);

% Helper Functions
function y = precision(M)
    % Input is a confusion matrix
    y = diag(M) ./ sum(M,2);
end

function y = recall(M)
    % Input is a confusion matrix
    y = diag(M) ./ sum(M,1);
end
