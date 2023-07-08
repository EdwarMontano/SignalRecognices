clear all; close all;clc;

% load('iris_dataset');
meas = csvread('../Dataset/Train_Input.csv');
species = csvread('../Dataset/Train_Targets.csv');
X = meas;
Y = species;

N_datos = 100;
c1 = 1; c2=2;
X1=[meas(1:N_datos,c1)' meas(101:200,c1)' meas(201:300,c1)'];
X2=[meas(1:N_datos,c2)' meas(101:200,c2)' meas(201:300,c2)'];
data =[X1;X2];
class = species;
Class = species;
scatter(X1,X2,N_datos,Class,'marker','*');
hold on;grid on;
title('Datos Originales');

% matriz de covarianza
C=cov(data');

X=data';

if ~exist('epsilon','var')
    epsilon = 0.001;
end

mu = mean(X); 
X = bsxfun(@minus, X, mu);
A = X'*X;
[V,D,notused] = svd(A);
whMat = sqrt(size(X,1)-1)*V*sqrtm(inv(D + eye(size(D))*epsilon))*V';
Xwh = (X*whMat)';  

cov(Xwh')    %permite verificar que la matriz de covarianza ahora es la identidad

% invMat = pinv(whMat);

figure;
scatter(Xwh(1,:),Xwh(2,:),N_datos,Class,'marker','o');
hold on;grid on;
title('Datos con Normalizacion Whitening');


% Clasificacin
tree4 = fitctree(Xwh', species);
% [dtnum,dtnode,dtclass] = treeval(tree4, Xwh);
% bad4 = ~strcmp(dtclass,species);
% error4=sum(bad4)/1.5;
% disp(strcat('Porcentaje de error =',num2str(error4),'% whitenning'))