% Maximum Likelihood Parameter Estimation of Gaussian pdfs

% One problem oftenmet in practice is that the pdfs describing the statistical distributionof the data in the
% classes are not known and must be estimated using the training data set.

clear all; close all; clc;

% La distribuci�n real de los datos es una Gaussiana con par�metros:
m1=[2 -2]';
S = [0.9 0.2; 0.2 0.3];

% Se generan datos aleatorios con la distribuci�n real:
randn('seed',0);
m = [2 -2]', S = [0.9 0.2; 0.2 .3],
N = 50;
% N = 5000; % se puede verificar que entre mayor sea el n�mero de muestras se acerca m�s al valor real de m y S
X = mvnrnd(m,S,N)';

% Como los par�metros de la distribuci�n son desconocidos, se calculan de
% los datos en X
m_calculada = mean(X')',
S_calculada= cov(X'),

