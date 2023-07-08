
% rbfEjemplo1.m
clear all; close all; clc;

x = [0 0 1 1;
0 1 0 1];
y = [0 1 1 0];
red = newrb(x,y,0.01,0.5);
yred = sim(red,x);
% disp('Presione tecla para continuar');
% pause;
close all;
% Cual es el proceso paso a paso?
l = size(x,2);
% Procesamiento en la capa oculta RBF
a = radbas(netprod(dist(red.IW{1},x),concur(red.b{1},l)));
disp('Resultado a la salida de la capa Oculta');
disp(a); % Observe en la ventana de comandos la salida en la capa RBF
% Procesamiento en la capa de salida
yobt = red.LW{2}*a+red.b{2};
disp('Resultado de verificacion de la red RBF');
disp(yobt); % Observe en la ventana de comandos la salida de la red RBF
% El siguiente código presenta las figuras de salida de la capa oculta de
% la red de base radial teniendo en cuenta conjuntos de puntos que van
% desde -0.5 hasta 1.5. Recuerde que este es el resultado de entrenar una
% red con solo 4 patrones.
X = -0.5:0.05:1.5;
L = length(X);
X1 = X(ones(L,1),:);
X2 = X1';
XT = [X1(:) X2(:)]';
A = radbas(netprod(dist(red.IW{1},XT),concur(red.b{1},L*L)));
NR = size(A,1); % Numero de funciones de base radial
Z = cell(NR,1);
for i=1:NR,
Z{i} = reshape(A(i,:),L,L);
end,
% Las figuras se presentan separadas dado que cada función de base radial
% responde a los estimulos de cada patrón de manera independiente.
figure,
for i=1:NR,
surf(X1,X2,Z{i});
hold on,
end,
hold off;
% La verificacion de la salida de la red RBF se presenta con la siguiente
% figura.
figure,
YO = red.LW{2}*A+red.b{2};
Y = reshape(YO,L,L);
surf(X1,X2,Y);