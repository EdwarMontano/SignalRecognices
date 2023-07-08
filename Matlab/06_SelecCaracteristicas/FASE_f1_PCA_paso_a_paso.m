clear all; close all; clc;

X= [ 2 2 3 4 5 5; 2 3 4 3 4 5]';

media = mean (X);
covarianza = cov(X);
[W,L] = eigs(covarianza);
lamda =diag(L);

Y = (X*W);

