clear all; close all;clc;

% load('iris_dataset');
load fisheriris;

N_datos = 50;
c1 = 3; c2=4;
X1=[meas(1:N_datos,c1)' meas(51:50+N_datos,c1)' meas(101:100+N_datos,c1)'];
X2=[meas(1:N_datos,c2)' meas(51:50+N_datos,c2)' meas(101:100+N_datos,c2)'];
data =[X1;X2];
class = ones(1,150);class(1,51:100)=2;class(1,101:end)=3;
Class = [class(1,1:N_datos) class(1,51:50+N_datos) class(1,101:100+N_datos)];
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