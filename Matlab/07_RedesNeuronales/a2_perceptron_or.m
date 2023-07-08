clear all; clc; close all;

x=[0 0 1 1; ...
   0 1 0 1];
yd=[0 1 1 1];
plotpv(x,yd);
red=newp(x,yd);

% red.iw{1,1}=[1 1];
% red.b{1}=0.5;
Pesos=red.iw{1,1};
Bias=red.b{1};
plotpc(Pesos,Bias)
title('Datos OR y Recta con los valores iniciales de pesos y bias');
pause

red=train(red,x,yd);

Pesos_f=red.iw{1,1};
Bias_f=red.b{1};
figure(1);
plotpc(Pesos_f,Bias_f);
title('Datos OR y Recta con los valores de pesos y bias despues del entrenamiento');
pause;

% verificacion de las respuestas
disp(strcat(' la respuesta ante [0; 0] es ',num2str(sim(red, [0; 0]))));
disp(strcat(' la respuesta ante [0; 1] es ',num2str(sim(red, [0; 1]))));
disp(strcat(' la respuesta ante [1; 0] es ',num2str(sim(red, [1; 0]))));
disp(strcat(' la respuesta ante [1; 1] es ',num2str(sim(red, [1; 1]))));
