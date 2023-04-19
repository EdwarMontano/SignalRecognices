% The Malanobish Distance Classifier

clear all; close all; clc;

% • The classes are equiprobable.
% • The data in all classes follow Gaussian distributions.
% • The covariance matrix is the same for all classes.

% • The covariancematrix is NOT a diagonal and all elements across the diagonal are equal.
S=[0.8 0.01 0.01;0.01 0.2 0.01; 0.01 0.01 0.2];

m1=[0 0 0]'; m2=[0.5 0.5 0.5]';
m=[m1 m2];

% Under these assumptions, it turns out that the optimal Bayesian classifier is equivalent to the minimum
% Euclidean distance classifier. That is, given an unknown x, assign it to class ?i if

x=[0.1 0.5 0.1]';

pc1 = sqrt((x-m1)'*inv(S)*(x-m1))
pc2 = sqrt((x-m2)'*inv(S)*(x-m2))

% Toma de decisión
if pc1>pc2
    disp('pertenece a la clase 1');
else
    disp('pertenece a la clase 2');
end