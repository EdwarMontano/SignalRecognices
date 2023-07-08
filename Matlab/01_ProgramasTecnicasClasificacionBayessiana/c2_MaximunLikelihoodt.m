% Maximum Likelihood Parameter Estimation of Gaussian pdfs

% One problem oftenmet in practice is that the pdfs describing the statistical distributionof the data in the
% classes are not known and must be estimated using the training data set.

clear all; close all; clc;

% La distribución real de los datos es una Gaussiana con parámetros:
num_clases = 3;
m1=[0 0 0]'; m2=[1 2 2]'; m3=[3 3 4]';
S1 = 0.8*eye(3);S2=S1;S3=S1;
% S1=[0.8 0.2 0.1; 0.2 0.8 0.2; 0.1 0.2 0.8];S2=S1;S3=S1;
% S1=[0.8 0.2 0.1; 0.2 0.8 0.2; 0.1 0.2 0.8];S2=[0.6 0.01 0.01; 0.01 0.8 0.01; 0.01 0.01 0.6];S3=[0.6 0.1 0.1;0.1 0.6 0.1; 0.1 0.1 0.6];
p1=1/3;p2=1/3;p3=1/3;

% Se generan los datos aleatorios con la distribución real para cada clase:
randn('seed',0);
N = 100;
% N = 5000; % se puede verificar que entre mayor sea el número de muestras se acerca más al valor real de m y S
X1 = mvnrnd(m1,S1,N)';
X2 = mvnrnd(m2,S2,N)';
X3 = mvnrnd(m3,S3,N)';

% Como los parámetros de la distribución son desconocidos, se calculan de
% los datos
m1_calculado = mean(X1')';
S1_calculado= cov(X1');
m2_calculado = mean(X2')';
S2_calculado= cov(X2');
m3_calculado = mean(X3')';
S3_calculado= cov(X3');

% Se genera un nuevo dato de entrada aleatorio para ser clasificado
% x = mvnrnd(m1,S1,1)';
% x = mvnrnd(m2,S2,1)';
x = mvnrnd(m3,S3,1)';

% Se verifica la probabilidad de que el dato en x1 pertenezca a la clase 1
constante1= (1/(((2*pi)^(num_clases/2))*(sqrt(det(S1_calculado)))));
pg1 = constante1*exp((-0.5)*(x-m1_calculado)'*inv(S1_calculado)*(x-m1_calculado));
pc1= p1*pg1;

% Se verifica la probabilidad de que el dato en x1 pertenezca a la clase 2
constante2= (1/(((2*pi)^(num_clases/2))*(sqrt(det(S2_calculado)))));
pg2 = constante2*exp((-0.5)*(x-m2_calculado)'*inv(S2_calculado)*(x-m2_calculado));
pc2= p2*pg2;

% Se verifica la probabilidad de que el dato en x1 pertenezca a la clase 3
constante3= (1/(((2*pi)^(num_clases/2))*(sqrt(det(S3_calculado)))));
pg3 = constante3*exp((-0.5)*(x-m3_calculado)'*inv(S3_calculado)*(x-m3_calculado));
pc3= p3*pg3;

p = [pc1,pc2,pc3]
[p_max,c] = max(p);
% Toma de decisión
disp(strcat('pertenece a la clase ',num2str(c)));
