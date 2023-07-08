clear all; close all; clc;

% El perceptrón tiene aprendizaje supervisado, es decir, se conoce la
% salida deseada ante una entrada. Asuma para el ejemplo el caso de una
% compuerta AND u OR

X = [0, 0, 1, 1; 0, 1, 0, 1];
% Yd = [0, 0, 0, 1]; % AND
Yd = [0, 1, 1, 1]; % OR

% Se grafican los puntos del ejemplo
figure(1);
plotpv(X,Yd);

% Como la salida deseada requiere solo 2 "estados", basta con el uso de una
% única neurona, y por tanto, sólo se modificará durante el aprendizaje un
% peso W11

% Paso 1: Iniciar aleatorialmente los pesos y bias
num_entradas = 2;
W = rand(num_entradas,1); B = rand(1,1); % iniciar los pesos con valores aleatorios
% W = [0.3; -0.2]; B = -0.6; % para que coincida con el ejemplo de la diapositiva

figure(1);
hold on;
m=W(1)/W(2);% La pendiente corresponde a la división de W1/W2  
b=B/W(2); % El bias permite un desplazamiento de la recta
x1=-0.2:0.2:1.2; % Intersección de la recta con el eje X1
x2=-m*x1-b; % Intersección de la recta con el eje X2
plot(x1,x2,'k'); % Se grafica la línea de separación
% También se puede usar plotpc([W1 W2],B)
title('Recta de Clasificación Inicial');
pause(1);

% Se define el número de iteraciones máximo para lograr el aprendizaje
num_iter = 20;
iteracion = 1;
error_deseado = 0;
num_ejemplos = length(Yd);
factor_aprendizaje = 0.5;
while (iteracion < num_iter)
    % Paso 2: se presentam los ejemplos de entrenamiento uno a uno. En este
    % caso son 4
    neta=W'*X+B*ones(1,num_ejemplos);
    % suponiendo función de transferencia bipolar
    Yred = escalon(neta);
    error=sum(((Yd-Yred).^2)); % calculo del error global
    if error ~= 0
        % Actualizar los pesos y bias
        W=(W'+factor_aprendizaje*(Yd-Yred)*X')';
        B=B+factor_aprendizaje*(Yd-Yred)*ones(4,1);
        % Grafica de la nueva recta despues del entrenamiento
        hold on;
        m=W(1)/W(2);% La pendiente corresponde a la división de W1/W2  
        b=B/W(2); % El bias permite un desplazamiento de la recta
        x1=-0.2:0.2:1.2; % Intersección de la recta con el eje X1
        x2=-m*x1-b; % Intersección de la recta con el eje X2
        plot(x1,x2,'k'); % Se grafica la línea de separación
        title(strcat('Clasificación en la iteración = ',num2str(iteracion)));
        pause(3);
        iteracion = iteracion+1;
    else
        msgbox('Error alcanzado');
        hold on;
        plot(x1,x2,'g'); % Se grafica la línea de separación
        iteracion = num_iter;
    end
end
figure;
plotwb([W; B]);

