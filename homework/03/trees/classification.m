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

X = adult_data(:,{'age', 'work_class', 'education_num', 'marital_status',...
    'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week',...
    'salary'});
% summary(X)

% =========================================================================
% 1.Use Matlab functions to construct decision trees over the dataset using 4-fold
% cross-validation. Briefly describe in your report the functions that you use and
% their parameters. Run at least 5 different experiments varying parameter values.
% Repeat the same experiments but now using pruning. Show the results of all of your
% experiments neatly organized on a table showing parameter values, classification
% accuracy, size of the tree (number of nodes and/or number of leaves), and runtime.

tree = fitctree(X, 'salary', 'CrossVal', 'on', 'Kfold', 4);
label = kfoldPredict(tree);
% confusion_train = confusionmat(adult_data.salary, label);
% confusion_train
kfoldLoss(tree)

tree_prune = fitctree(X, 'salary', 'CrossVal', 'on', 'KFold', 4, 'Prune', 'on');
label_prune = kfoldPredict(tree_prune);
kfoldLoss(tree_prune)

% =========================================================================
% 2. Select the pruned tree with smallest size. Use Matlab plotting functions
% to depict the tree. Include the plot in your report (or at least the top
% levels if the tree is too large). Briefly comment on any interesting aspect
% of this tree.

% view(model.Trained{1},'Mode','graph')

% =========================================================================
% 3. Random forest technique. Run at least 5 different experiments varying
% parameter values. Show the results of your experiments neatly organized on
% a table showing parameter values, classification accuracy, size of the
% random forest, and runtime.

% https://www.mathworks.com/help/stats/fitcensemble.html
random_forest = fitcensemble(adult_data, 'salary');
label_rf = predict(random_forest, adult_data);
resubLoss(random_forest)
