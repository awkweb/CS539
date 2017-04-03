% =========================================================================
% SUPPORT VECTOR MACHINES
rng(1);

% =========================================================================
% Load train data
filename = 'data/adult/adult.dat';
adult_data = readtable(filename);
adult_data.Properties.VariableNames = {'age', 'work_class', 'fnlwgt',...
    'education', 'education_num', 'marital_status', 'occupation',...
    'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',...
    'hours_per_week', 'native_country','salary'};

% =========================================================================
% 1. Use Matlab functions to construct a support vector machine over the
% dataset using 4-fold cross-validation. Briefly describe in your report the
% functions that you use and their parameters.

% Split data into inputs and target
cvp_indices = cvpartition(height(adult_data), 'holdout', 0.25);
adult_data_cvp = adult_data(training(cvp_indices),:);
X = adult_data_cvp(:,{'age', 'work_class', 'education_num', 'marital_status',...
                  'race', 'sex', 'capital_gain', 'capital_loss',...
                  'hours_per_week'});
Y = adult_data_cvp(:,{'salary'});

svm = fitcsvm(X, Y, 'KernelFunction', 'linear');

% cv_svm = crossval(svm, 'Kfold', 4);
time_start = tic;
label = predict(svm);
% [label, score] = kfoldPredict(cv_svm);
time_elapsed = toc(time_start);
classification_error = kfoldLoss(cv_svm);
classification_accuracy = 1 - classification_error;

display(cv_svm);
% view(cv_tree.Trained{1}, 'Mode', 'graph');
display(time_elapsed);
display(classification_error);
display(classification_accuracy);

% =========================================================================
% 2. Run at least 12 different experiments varying parameter values for each
% of the following kernel functions (run at least 4 experiments for each one
% of the 3 kernel functions required): 
%    - polynomial (including linear, quadratic, ...)
%    - radial-basis functions (Gaussian)
%    - sigmoid (tanh)
% Show the results of all of your experiments neatly organized on a table
% showing kernel function used, parameter values, classification accuracy,
% and runtime.


% =========================================================================
% 3. Pick the experiment that you think produced the best result. Justify
% your choice in your report. Use Matlab functionality to plot a 2 or 3
% dimensional depiction of data instances in each of the two classes, support
% vectors, and the decision boundary.



