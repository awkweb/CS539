% =========================================================================
% NAIVE BAYES
rng(1);

% =========================================================================
% Load data
filename = './data/adult/adult.dat';
adult_data = readtable(filename);
adult_data.Properties.VariableNames = {'age', 'work_class', 'fnlwgt',...
    'education', 'education_num', 'marital_status', 'occupation',...
    'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',...
    'hours_per_week', 'native_country', 'salary'};

% =========================================================================
% 0. Split dataset into 2 parts: 75% for training and 25% for testing.
cvp_indices = cvpartition(height(adult_data), 'holdout', 0.25);
adult_data_train = adult_data(training(cvp_indices),:);
X_train = adult_data_train(:,{'age', 'work_class', 'education_num', 'marital_status',...
                  'race', 'sex', 'capital_gain', 'capital_loss',...
                  'hours_per_week'});
Y_train = adult_data_train(:,{'salary'});

adult_data_test = adult_data(test(cvp_indices),:);
X_test = adult_data_test(:,{'age', 'work_class', 'education_num', 'marital_status',...
                  'race', 'sex', 'capital_gain', 'capital_loss',...
                  'hours_per_week'});
Y_test = adult_data_test(:,{'salary'});

% =========================================================================
% 1. Create a Naive Bayes model over the training dataset. Look at the
% conditional probability tables and select one that looks interesting.
% Include it in your report and explain why you think it is interesting.
nb = fitcnb(X_train, Y_train);

cv_nb = crossval(nb, 'Kfold', 4);
time_start = tic;
[label, score] = kfoldPredict(cv_nb);
time_elapsed = toc(time_start);
% [post,cpre1,logp] = posterior(nb, adult_data);

% =========================================================================
% 2. Classify the data instances in the test dataset using this Naive Bayes
% model. Include in your report the accuracy, precision, and recall values
% obtained.
classification_error = kfoldLoss(cv_nb);
classification_accuracy = 1 - classification_error;

display(cv_nb);
display(time_elapsed);
display(classification_error);
display(classification_accuracy);

confusion_train = confusionmat(Y_train.salary, label);
precision_train = mean(precision(confusion_train));
recall_train = mean(recall(confusion_train));
display(precision_train);
display(recall_train);

function y = precision(M)
    y = diag(M) ./ sum(M,2);
end
 
function y = recall(M)
    y = diag(M) ./ sum(M,1)';
end
