close all; clear all;clc;

x=0:0.2:2*pi;
y=sin(x)+cos(2*x);
figure;
fplot('sin(x)+cos(2*x)',[0 2*pi]);
title('Funcion a Aprender')
% Visualización de la función de base radial
xr = -3.5:0.1:3.5;
yr = radbas(xr);
figure;
plot(xr,yr)
title('Funcion de Activacion Usada por la Red');
% Creación de la red
red = newrb(x,y,1E-4,1);
X = 0:0.1:2*pi;
Yred= sim(red,X);
figure;
plot(x,y,'ok','MarkerFaceColor','b');
hold on;
plot(X,Yred,'or');
title('Azul = Original, Rojo = Red RBF');
grid on;
hold off;