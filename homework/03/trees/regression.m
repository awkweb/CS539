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

% =========================================================================
% 1. Use Matlab functions to construct regression trees over the dataset using
% 4-fold cross-validation. Briefly describe in your report the functions that
% you use and their parameters. Run at least 5 different experiments varying
% parameter values. Repeat the same experiments but now using pruning. Show
% the results of all of your experiments neatly organized on a table showing
% parameter values, Sum of Square Errors (SSE), Root Mean Square Error (RMSE),
% Relative Square Error (RSE), Coeffient of Determination (R2), size of the
% tree (number of nodes and/or number of leaves), and runtime.

% Split data into inputs and target
cvp_indices = cvpartition(height(adult_data), 'holdout', 0.25);
adult_data_cvp = adult_data(training(cvp_indices),:);
X = adult_data_cvp(:,{'age', 'work_class', 'marital_status', 'race',...
                      'sex', 'capital_gain', 'capital_loss',...
                      'hours_per_week', 'salary'});
Y = adult_data_cvp(:,{'education_num'});

tree = fitrtree(X, Y, 'MaxNumSplits', 20, 'MinLeafSize', 8,...
                'PredictorSelection', 'curvature', 'Surrogate', 'on');
cv_tree = crossval(tree, 'Kfold', 4);
time_start = tic;
label = kfoldPredict(cv_tree);
time_elapsed = toc(time_start);

display(cv_tree);
display(time_elapsed);

sse = mean(kfoldfun(cv_tree, @sseEval));
rmse = mean(kfoldfun(cv_tree, @rmseEval));
rse = mean(kfoldfun(cv_tree, @rseEval));
r2 = mean(kfoldfun(cv_tree, @r2Eval));

display(sse);
display(rmse);
display(rse);
display(r2);

% =========================================================================
% Sum of Square Errors (SSE), Root Mean Square Error (RMSE),
% Relative Square Error (RSE), Coeffient of Determination (R2)
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
