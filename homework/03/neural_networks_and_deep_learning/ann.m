% =========================================================================
% ARTIFICIAL NEURAL NETWORK
% http://web.cs.wpi.edu/~cs539/s17/HW/HW3/
% https://www.mathworks.com/help/nnet/examples/create-simple-deep-learning-network-for-classification.html
rng(1);

% =========================================================================
% Load train data
filename = 'data/optdigits/adult.dat';
adult_data = readtable(filename);
adult_data.Properties.VariableNames = {'age', 'work_class', 'fnlwgt',...
    'education', 'education_num', 'marital_status', 'occupation',...
    'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',...
    'hours_per_week', 'native_country','salary'};

X = adult_data(:,{'age', 'work_class', 'education_num', 'marital_status',...
    'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week',...
    'salary'});


