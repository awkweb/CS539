% Thomas Meagher

% a. Generate 100 sequences
rng(1);

TRANS = [
    0.3 0.5 0.2;...
    0.2 0.2 0.6;...
    0.7 0.1 0.2];

EMIS = [
    1, 0, 0;...
    0, 1, 0;...
    0, 0, 1;];

[seq, states] = hmmgenerate(100, TRANS, EMIS);
disp(mean(seq));
disp(mean(states));

% b. Estimate initial probabilities
p = [ 0.4, 0.1, 0.5 ];

TRANS_HAT = [ 0 p; zeros(size(TRANS, 1), 1) TRANS ];
EMIS_HAT = [ zeros(1, size(EMIS, 2)); EMIS ];

[seq_HAT, states_HAT] = hmmgenerate(100, TRANS_HAT, EMIS_HAT);

disp(mean(seq_HAT));
disp(mean(states_HAT));
