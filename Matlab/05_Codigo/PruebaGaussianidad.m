clear all; close all;clc;

media = 10; desviacion_estandar=0.5;
pd = makedist('Normal',media,desviacion_estandar);
t = [media-(5*desviacion_estandar):desviacion_estandar/10:media+(5*desviacion_estandar)];
x = normpdf(t,media,desviacion_estandar);
% x = pdf('Normal',t,media,desviacion_estandar);
% x = = pdf('Exponential',t,media);
% x = x+(max(x)/100)*randn(size(x));
plot(t,x);grid on;
figure; violin(x');
% x = randn(200,1);
nivel_significancia = 5/100; % valores entre 0 y 1
[h, p, k, c] = lillietest(x,'Distr','norm','Alpha',nivel_significancia);

if h==0
    msgbox(strcat('Es una funcion gausiana, p > significancia, ',num2str(p),'>',num2str(nivel_significancia)));
else
    msgbox(strcat('NO es una funcion gausiana, p < significancia, ',num2str(p),'<',num2str(nivel_significancia)));
end