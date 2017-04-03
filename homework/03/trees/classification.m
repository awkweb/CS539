% =========================================================================
% CLASSIFICATION TREES
rng(1);

% =========================================================================
% Load train data
filename = 'data/adult/adult.dat';
adult_data = readtable(filename);
adult_data.Properties.VariableNames = {'age', 'work_class', 'fnlwgt',...
    'education', 'education_num', 'marital_status', 'occupation',...
    'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',...
    'hours_per_week', 'native_country','salary'};
% summary(adult_data)

% =========================================================================
% 1.Use Matlab functions to construct decision trees over the dataset using 4-fold
% cross-validation. Briefly describe in your report the functions that you use and
% their parameters. Run at least 5 different experiments varying parameter values.
% Repeat the same experiments but now using pruning. Show the results of all of your
% experiments neatly organized on a table showing parameter values, classification
% accuracy, size of the tree (number of nodes and/or number of leaves), and runtime.

% Split data into inputs and target
cvp_indices = cvpartition(height(adult_data), 'holdout', 0.25);
adult_data_cvp = adult_data(training(cvp_indices),:);
X = adult_data_cvp(:,{'age', 'work_class', 'education_num', 'marital_status',...
                  'race', 'sex', 'capital_gain', 'capital_loss',...
                  'hours_per_week'});
Y = adult_data_cvp(:,{'salary'});

% Fit k-fold cross-validated classification tree with max number of splits
% Grow unbiased trees by using the curvature test for PredictorSelection
% Missing observations in the data ==> Surrogate splits on
tree = fitctree(X, Y, 'MaxNumSplits', 20, 'MinLeafSize', 8,...
                'PredictorSelection', 'curvature', 'Surrogate', 'on');

cv_tree = crossval(tree, 'Kfold', 4);
time_start = tic;
[label, score] = kfoldPredict(cv_tree);
time_elapsed = toc(time_start);
classification_error = kfoldLoss(cv_tree);
classification_accuracy = 1 - classification_error;

display(cv_tree);
% view(cv_tree.Trained{1}, 'Mode', 'graph');
display(time_elapsed);
display(classification_error);
display(classification_accuracy);

% Create confusion matrix for classification tree
confusion_train = confusionmat(Y.salary, label);
display(confusion_train);

% =========================================================================
% 2. Select the pruned tree with smallest size. Use Matlab plotting functions
% to depict the tree. Include the plot in your report (or at least the top
% levels if the tree is too large). Briefly comment on any interesting aspect
% of this tree.

tree_pruned = prune(tree, 'level', 4);
% view(tree_pruned, 'Mode', 'graph');

% =========================================================================
% 3. Random forest technique. Run at least 5 different experiments varying
% parameter values. Show the results of your experiments neatly organized on
% a table showing parameter values, classification accuracy, size of the
% random forest, and runtime.
% https://www.mathworks.com/help/stats/fitcensemble.html

random_forest = fitcensemble(X, Y);
% random_forest_2 = fitcensemble(adultdata,'salary ~ age + education');
time_start = tic;
label_rf = predict(random_forest, X);
time_elapsed = toc(time_start);
classification_error = resubLoss(random_forest);
classification_accuracy = 1 - classification_error;

display(random_forest);
display(time_elapsed);
display(classification_error);
display(classification_accuracy);

confusion_train = confusionmat(Y.salary, label_rf);
display(confusion_train);

% =========================================================================
% Misc

% imp = predictorImportance(tree);
% figure;
% bar(imp);
% title('Predictor Importance Estimates');
% ylabel('Estimates');
% xlabel('Predictors');
% h = gca;
% h.XTickLabel = tree.PredictorNames;
% h.XTickLabelRotation = 45;
% h.TickLabelInterpreter = 'none';

% tree = fitctree(X, Y, 'OptimizeHyperparameters', 'auto', 'Kfold', 4);

% Functions to generate precision and recall of tree
% precision_train = mean(precision(confusion_train));
% recall_train = mean(recall(confusion_train));
% 
% display(precision_train);
% display(recall_train);

% function y = precision(M)
%   y = diag(M) ./ sum(M,2);
% end
% 
% function y = recall(M)
%   y = diag(M) ./ sum(M,1)';
% end
