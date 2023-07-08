clear all; close all; clc;

load fisheriris;

X=meas(51:150,:)';
BC=zeros(1,100);BC(1,51:end)=1;
I = rankfeatures(X,BC,'Criterion','entropy','NumberOfIndices',4)  % Entropa, asume gaussianidad
% I = rankfeatures(X,BC,'Criterion','bhattacharyya','NumberOfIndices',4) % Bhattacharryya, asume gausianidad
% I = rankfeatures(X,BC,'NumberOfIndices',4)  % ttest, asume gaussianidad
% I = rankfeatures(X,BC,'Criterion','roc','NumberOfIndices',4)  % roc, independiente de la pdf
% I = rankfeatures(X,BC,'Criterion','wilcoxon','NumberOfIndices',4)  % Wilcoxon, no asume gaussianidad

C = classify(X(I,:)',X(I,:)',double(BC)); % anlisis discriminante
cp = classperf(BC,C);
cp.CorrectRate

