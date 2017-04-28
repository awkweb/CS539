% =========================================================================
% BAYESIAN NETWORKS
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
X_test = adult_data_train(:,{'age', 'work_class', 'education_num', 'marital_status',...
                  'race', 'sex', 'capital_gain', 'capital_loss',...
                  'hours_per_week'});
Y_test = adult_data_train(:,{'salary'});

% =========================================================================
% 2. Using Matlab functions, create a (non-Naive) Bayesian network over the
% training dataset. Plot the graphical model obtained. Look at the
% conditional probability tables and select one that looks interesting.
X = 1;
Q = 2;
Y = 3;
dag = zeros(3,3);
dag(X,[Q Y]) = 1;
dag(Q,Y) = 1;
ns = [1 2 2];
dnodes = [2];
bnet = mk_bnet(dag, ns, dnodes);

x = 0.5;
bnet.CPD{1} = root_CPD(bnet, 1, x);
bnet.CPD{2} = softmax_CPD(bnet, 2);
bnet.CPD{3} = gaussian_CPD(bnet, 3);

data_case = sample_bnet(bnet, 'evidence', {0.8, [], []})
ll = log_lik_complete(bnet, data_case)

data_case = sample_bnet(bnet, 'evidence', {-11, [], []})
ll = log_lik_complete(bnet, data_case)

% =========================================================================
% 3. Classify the data instances in the test dataset using this Bayesian
% Network. Include in your report the accuracy, precision, and recall values
% obtained.









