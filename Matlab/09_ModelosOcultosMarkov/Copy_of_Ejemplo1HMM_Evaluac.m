clc; clearvars; close all;

% Durante la Evaluacin se determina la PROBABILIDAD A POSTERIORI de que
% una secuencia de ESTADOS VISIBLES haya sido generada por el modelo HMM.
% Para Clasificar, se asignando el  vector de entrada a modelo HMM con
% la probabilidad ms alta

% El modelo HMM est definido por la matriz de probabilidades de transicin
% entre estados ocultos y la matriz de emisin de estados visibles 

% Ejemplo 1: 2 estados ocultos y 6 estados visibles
TRANS = [.9 .1; .05 .95];
EMIS = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6;...
7/12, 1/12, 1/12, 1/12, 1/12, 1/12];

% Ejemplo 2: 4 estados ocultos y 5 estados visibles
% TRANS= [1  0   0   0;   0.2  0.3  0.1   0.4;  0.2  0.5  0.2  0.1;  0.8  0.1  0   0.1];
% EMIS= [1  0  0  0  0;  0  0.3  0.4  0.1  0.2;  0  0.1  0.1  0.7  0.1;  0  0.5  0.2   0.1  0.2];

% Ejemplo 3: 4 estados ocultos y 5 estados visibles
% TRANS= [0.997  0.001   0.001   0.001;   0.2  0.3  0.1   0.4;  0.2  0.5  0.2  0.1;  0.8  0.1  0   0.1];
% EMIS= [0.996  0.001  0.001  0.001  0.001;  0.001  0.3  0.4  0.1  0.2;  0.001  0.1  0.1  0.7  0.1;  0.001  0.5  0.2   0.1  0.2];

% Ejemplo 4: 4 estados ocultos y 5 estados visibles
% TRANS= [0.001   0.001   0.001 0.997 ;   0.3  0.1   0.4 0.2;  0.5  0.2  0.1 0.2;  0.1  0   0.1 0.8];
% EMIS= [ 0.001  0.3  0.4  0.1  0.2;  0.001  0.1  0.1  0.7  0.1;  0.001  0.5  0.2   0.1  0.2; 0.996  0.001  0.001  0.001  0.001];

% Generar una secuencia aleatoria de estados visibles V
% (La funcin hmmgenerate() genera estados ocultos W y visibles, aunque 
% para la "Evaluacin" solo se utiliza la secuencia de estados visibles V) 
% Por defecto se asume que en el instante t, el estado oculto inicial es w1 

% Generar una secuencia estados Visibles por "defecto"
 Ti=10; %Longitud secuencia por "defecto"
 [Vi,Wi] = hmmgenerate(Ti,TRANS,EMIS); 
 
 [Neo,Nev]=size(EMIS); % Neo= nmero estados ocultos, Nev= Nmero estados visibles
 Vch = inputdlg({['Secuencia V separado por espacios (Hay ' num2str(Nev) ' diferentes)' ]},...
     'EVALUACIN HMM',[1 60],{num2str(Vi)});
 V=str2num(Vch{1}); % Secuencia V que se evaluar en el modelo HMM
 T=length(V); %Longitud de la secuencia V que se evaluar 
 
% EVALUACIN: Estimar la probabilidad de que una secuencIa V haya sido 
% generada por el modelo HMM
     
% Matriz ALPHAS contiene las Probabilidades condicionales de estar en el
% estado k en el paso i, dado la secuencia observada V.
ALPHAS = hmmdecode(V,TRANS,EMIS); 
%[ALPHAS,logpseq,FORWARD,BACKWARD,S]=hmmdecode(V,TRANS,EMIS);

% Calcular ndices de la probabilidad final (Probabilidad condicional de que 
% el modelo THETA genere la secuencia V). Se selecciona la mxima.

[Prob_m,Index_m]=max(ALPHAS);
P_V_THETA=ALPHAS(Index_m(T),T);


% Visualizacin
 Mensaje1=['Secuencia a Evaluar' '\n' 'V = ' num2str(V) ' \n'  ...
 '\n' 'Probabilidad de que el modelo genere la secuencia V' '\n'  'P_V_Theta = ' num2str(P_V_THETA)];   
 helpdlg(sprintf(Mensaje1),'EVALUACIN');

       















