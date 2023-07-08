clc; clearvars; close all;

% Para la DECODIFICACIN Si se conoce el modelo "real"; es decir, las matrices 
% de emisin y transicin de estados.
% La Decodifacin entrega una secuencia de estados ocultos ms probable que
% haya generado un secuencia de estados visibles.

% Matriz de probabilidades de transicin y emisin del modelo

TRANS = [.5 .25 .25; .25 .5 .25; .25 .5 .25];
EMIS = [1, 0, 0;...
        0, 1, 0;...
        0, 0, 1];

% *************************************************************
% Generar una secuencia aleatoria de estados y emisiones
% Por defecto el estado oculto inicial es w1 (Puede cambiarse)

% Especificar la longitud de la secuencia de estados visibles y ocultos para entrenamiento:  V_Train, W_Train
 LT = inputdlg({'Longitud Secuencia, T='},'Generar Secuencia',[1 30],{'20'});
 T=str2num(LT{1}); % Longitud de la secuencia de estados emitidos V.

% Generar Secuencias de estados ocultos y visibles para Decodificar
% La secuencia V se utiliza para decodificar
% La secuencia W se utiliza para evaluar la precisin de la estimacin
V = csvread('../Dataset/Train_InputTHMM.csv');
W=zeros(1,300);
W(1,1:100)=1;
W(1,101:200)=2;
W(1,201:300)=3;
% [V,W] = hmmgenerate(T,TRANS,EMIS); % 


% DECODIFICACIN
% ESTIMAR las secuencia de ESTADOS OCULTOS W_estimad que gener la secuencia 
% de ESTADOS VISIBLES V
% El algoritmo de Viterbi calcula la secuencia  W_estimad ms probable.

W_estimad = hmmviterbi(V, TRANS, EMIS);

% Clculo del porcentaje de acierto
Porcentaje_Acierto=100*sum(W==W_estimad)/T;

% Visualizacin
Mensaje1=['SECUENCIAS A PARTIR DEL MODELO REAL' '\n' '\n' 'Secuencia Estados Ocultos' '\n'   'W = ' num2str(W) '\n' '\n' 'Secuencia Estados Visibles' '\n' 'V= ' num2str(V) ...
'\n''\n' 'SECUENCIA ESTIMADA' '\n' '\n' 'Secuencia Estados Ocultos ESTIMADA' '\n'  'W^= ' num2str(W_estimad) '\n' ...
 '\n' 'Porcentaje ACIERTO = ' num2str(Porcentaje_Acierto)];

helpdlg(sprintf(Mensaje1),'DECODIFICACIN');
        
  











