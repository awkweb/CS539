% =========================================================================
% REGRESSION TREES
rng(1);

% =========================================================================
% Load train data
filename = 'data/adult/adult.dat';
adult_data = readtable(filename);
adult_data.Properties.VariableNames = {'age', 'work_class', 'fnlwgt',...
    'education', 'education_num', 'marital_status', 'occupation',...
    'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',...
    'hours_per_week', 'native_country','salary'};

X = adult_data(:,{'age', 'work_class', 'education_num', 'marital_status',...
    'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week',...
    'salary'});

% =========================================================================
% 1. Use Matlab functions to construct regression trees over the dataset using
% 4-fold cross-validation. Briefly describe in your report the functions that
% you use and their parameters. Run at least 5 different experiments varying
% parameter values. Repeat the same experiments but now using pruning. Show
% the results of all of your experiments neatly organized on a table showing
% parameter values, Sum of Square Errors (SSE), Root Mean Square Error (RMSE),
% Relative Square Error (RSE), Coeffient of Determination (R2), size of the
% tree (number of nodes and/or number of leaves), and runtime.

tree = fitrtree(X, 'education_num', 'CrossVal', 'on', 'Kfold', 4);
label = kfoldPredict(tree);
kfoldLoss(tree)

tree_prune = fitrtree(X, 'education_num', 'CrossVal', 'on', 'KFold', 4, 'Prune', 'on');
label_prune = kfoldPredict(tree_prune);
kfoldLoss(tree_prune)

% =========================================================================
% 2. Select the pruned tree with smallest size. Use Matlab plotting functions
% to depict the tree. Include the plot in your report (or at least the top
% levels if the tree is too large). Briefly comment on any interesting aspect
% of this tree.
