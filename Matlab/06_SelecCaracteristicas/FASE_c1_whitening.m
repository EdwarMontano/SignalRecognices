clear all; close all;clc;


gaussian = struct('sigma', [1 0.8; 0.8 1], 'median', [2 4.5]);
% numero de puntos 
N = 400;
% vector de datos
data = zeros(2, N);

for i=1:N
    data(:,i) = gaussian.median + (gaussian.sigma * [randn; randn])';
end

% plot
figure
plot(data(1,:), data(2,:), 'k+');
axis equal
xlabel('x'),ylabel('y');
grid on;

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
Xwh = X*whMat;  
hold on;
plot(Xwh(:,1), Xwh(:,2), 'r+');
title('Negro - Datos Originales, Rojo - Datos Normalización Whitening');

cov(Xwh)    %permite verificar que la matriz de covarianza ahora es la identidad

% invMat = pinv(whMat);

