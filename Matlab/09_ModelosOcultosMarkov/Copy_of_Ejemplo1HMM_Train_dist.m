clc; clearvars; close all;

% Durante el entrenamiento NO se conoce el modelo "real". Pero para efectos de crear un
% data set y para comparar la precisin del entrenamiento, es necesario
% partir de un modelo HHM "REAL" que consiste en la matriz de probabilidades de 
% transicin y emisin de estados del modelo

TRANS = [.5 .25 .25; .25 .5 .25; .25 .5 .25];
EMIS = [2/3, 1/3, 0;...
        0, 2/3, 1/3;...
        1/3, 0, 2/3];


% *************************************************************
% **********Procedimiento de Entrenamiento *************************
% Estima las matrices de transicin entre estados ocultos  y
% de emision de estados visibles.

TipoEnt= menu('Seleccione condicin','Conoce: W_Train y V_Train','Conoce: V_Train, Supuesto: TRANS_Supuesto y EMIS_Supuesto');
switch TipoEnt
    case 1
        % *********************************************************************
        % Entrenamiento a partir de la secuencia de estados ocultos visitados 
        % y la secuencia de estados visibles.

        % Especificar la longitud de la secuencia de estados visibles y ocultos para entrenamiento:  V_Train, W_Train
         LT = inputdlg({'Longitud Secuencia, T='},'Generar Secuencia',[1 30],{'20'});
         T=str2num(LT{1}); % Longitud de la secuencia de estados emitidos V.

        
        % Generar Secuencias de estados ocultos y visibles de entrenamiento       
        [V_Train,W_Train] = hmmgenerate(T,TRANS,EMIS); 

        % Entrenamiento 
        [TRANS_Estim, EMIS_Estim]=hmmestimate (V_Train, W_Train);    

              
        %Visualizacin 
        Mensaje1={'Matriz TRANSICION Real' num2str(TRANS) ' ' 'Matriz EMISION Real' num2str(EMIS) ' ' 'Datos ENTRENAMIENTO' ' ' 'W_Train ='  num2str(W_Train) '' 'V_Train= ' num2str(V_Train) ...
        ' ' 'Salida --> ESTIMACION MATRICES' ' ' 'Matriz TRANS^ = '  num2str(TRANS_Estim) '' 'Matriz EMIS^ = '  num2str(EMIS_Estim) ' '};   

        msgbox(Mensaje1,'ENTRENAMIENTO A');

    case 2
        % **************************************************************************
        % Entrenamiento a partir de la secuencia de estados visibles de entrenamiento 
        % y de supuestos sobre las matrices de transicin y de emisin.
        
        % Especificar la longitud de la secuencia de estados visibles y ocultos para entrenamiento:  V_Train, W_Train
         LT = inputdlg({'Longitud Secuencia, T='},'Generar Secuencia',[1 30],{'20'});
         T=str2num(LT{1}); % Longitud de la secuencia de estados emitidos V.

        % Generar Secuencias de estados ocultos y visibles de entrenamiento 
        [V_Train,W_Train] = hmmgenerate(T,TRANS,EMIS);
         
        % Valores Supuestos (o aproximaciones)  de  las matrices de transicin y de emisin.
        TRANS_Supuest=[0.8  0.2 ; 0.3  0.7];
        EMIS_Supuest= [1/6  1/3  0.0   1/3  0.0   1/6;...
                       1/5   0   2/5   0    2/5   0];

        
        %% Seleccin Algoritmo Entrenamiento
        
        TipoAlgor= menu('Algoritmo Entrenamiento','Viterbi','BaumWelch');
        switch TipoAlgor
            case 1 % Viterbi
                [TRANS_Estim, EMIS_Estim] = hmmtrain(V_Train, TRANS_Supuest, EMIS_Supuest, 'Algorithm','Viterbi');
            case 2 % BaumWelch
                [TRANS_Estim, EMIS_Estim] = hmmtrain(V_Train, TRANS_Supuest, EMIS_Supuest, 'Algorithm','BaumWelch'); % Por default 'Algorithm'= 'BaumWelch'
        end 

        %Visualizacin 
        Mensaje1={'Matriz TRANSICION Real' num2str(TRANS) ' ' 'Matriz EMISION Real' num2str(EMIS) ' ' 'Datos ENTRENAMIENTO' ' ' 'TRANS_Supuesta ='  num2str(TRANS_Supuest) ' ' 'EMIS_Supuest= ' num2str(EMIS_Supuest) ...
        '' 'V_Train' num2str(V_Train) ' '   'Salida --> ESTIMACION MATRICES' ' ' 'Matriz TRANS^ = '  num2str(TRANS_Estim) '' 'Matriz EMIS^ = '  num2str(EMIS_Estim) ' '};   

        msgbox(Mensaje1,'ENTRENAMIENTO B');

end
 
% Distancia (Frobenius) entre Matriz Transicinn Real y Estimada

dif_T=TRANS-TRANS_Estim;
dist_TRANS_frob=norm(dif_T(:),'fro')

% Distancia (Frobenius) entre Matriz Emisin Real y Estimada

dif_E=EMIS-EMIS_Estim;
dist_EMIS=norm(dif_E(:),'fro')

















