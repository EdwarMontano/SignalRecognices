% Maximum Likelihood Parameter Estimation of Gaussian pdfs

% One problem oftenmet in practice is that the pdfs describing the statistical distributionof the data in the
% classes are not known and must be estimated using the training data set.

clear all; close all; clc;

% La distribución real de los datos es una Gaussiana con parámetros:
m1=[2 -2]';
S = [0.9 0.2; 0.2 0.3];

% Se generan datos aleatorios con la distribución real:
randn('seed',0);
m = [2 -2]', S = [0.9 0.2; 0.2 .3],
N = 50;
% N = 5000; % se puede verificar que entre mayor sea el número de muestras se acerca más al valor real de m y S
X = mvnrnd(m,S,N)';

% Como los parámetros de la distribución son desconocidos, se calculan de
% los datos en X
m_calculada = mean(X')',
S_calculada= cov(X'),

