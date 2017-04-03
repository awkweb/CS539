% =========================================================================
% ARTIFICIAL NEURAL NETWORK
% http://web.cs.wpi.edu/~cs539/s17/HW/HW3/
% http://www.mathworks.com/help/nnet/ref/patternnet.html
rng(1);

% =========================================================================
% Load data
load data/optdigits/optdigits_train.dat
load data/optdigits/optdigits_test.dat

% =========================================================================
% 1. Classification using Artificial Neural Networks (ANNs):
% Use Matlab functions to construct and train ANNs over optdigits.tra
% and then test them over optdigits.tes.

inputs = optdigits_train(:,1:end-1);
inputs_2 = optdigits_train(:,1:end-1);
targets = optdigits_train(:,end);
net = patternnet(2);
net = train(net, inputs, targets);

% [x,t] = iris_dataset;
% net = patternnet(2);
% net = train(net,x,t);
% view(net)
% y = net(x);
% perf = perform(net,t,y);
% classes = vec2ind(y);