% KNN classifier

clearvars; close all; clc;

load('iris_dataset');

N =50;
N_train = 30;
num_clases = 3;
k = 4;
% distancia ='euclidean'
distancia ='mahalanobis'
class = ones(1,150);class(1,51:100)=2;class(1,101:end)=3;

% Gráfica de los puntos de entrenamiento (solo características 3 y 4)
X1=[irisInputs(3,1:N_train) irisInputs(3,51:51+N_train) irisInputs(3,101:101+N_train)]; %Característica 3
X2=[irisInputs(4,1:N_train) irisInputs(4,51:51+N_train) irisInputs(4,101:101+N_train)];  %Característica 4
X_train =[X1;X2];
C = [class(1,1:N_train) class(1,51:51+N_train) class(1,101:101+N_train)];

%scatter(X1,X2,40,C,'marker','d');
%scatter(X1,X2,30,C,'marker','*');
clr=[1 0 0; 0 0 1; 0 1 0]; %Clase 1= Rojo; Clase 2=azul   ; Clase3=verde  
sym=['d' 'd' 'd'];
siz=6;
og1=gscatter(X1,X2,C,clr,sym,siz,'off'); %Specifies the marker color clr, symbol sym, and size siz for each group.


hold on;grid on;

% Se crea un clasificador del tipo k vecinos
mdl = ClassificationKNN.fit(X_train',C','NumNeighbors',k)
md1.distance=distancia

% Se definen los puntos de validación
X1_val=[irisInputs(3,N_train+1:50) irisInputs(3,51+N_train:100) irisInputs(3,101+N_train:end)];
X2_val=[irisInputs(4,N_train+1:50) irisInputs(4,51+N_train:100) irisInputs(4,101+N_train:end)];
L_val = (50-N_train); %Cantidad de datos para validar
class_val =zeros(3,L_val*3);
class_val(1,1:L_val)=1;
class_val(2,L_val+1:L_val*2)=1;
class_val(3,2*L_val+1:end)=1;
class_vtrue =zeros(3,L_val*3);

for i=1:L_val*3  %Cantidad de datos a validar por el número de clases.
    X = [X1_val(i);X2_val(i)];
    % Algoritmo de k-neightbours
    [n,d] = knnsearch(X_train',X','k',k);
    % Se determina la clase
    [label,score,cost] = predict(mdl,X');    
    class_vtrue(label,i)=1;
    % Grafica el punto de validación en un color dependiendo de la clase
    % verdadera y rodeado de un circulo con un color de acuerdo con la
    % clase asignada
    [d2,j1]=max(class_val(:,i));
    switch j1
        case (1)
            og2= plot(X(1),X(2),'r*');
        case (2)
            og3=plot(X(1),X(2),'b*');
        otherwise
             og4=plot(X(1),X(2),'g*');
    end
    switch label
        case (1)
             og5=plot(X(1),X(2),'ro', 'markersize',10,'LineWidth',0.85);
        case (2)
             og6=plot(X(1),X(2),'bo', 'markersize',10,'LineWidth',0.85);
        otherwise
             og7=plot(X(1),X(2),'go','markersize',10,'LineWidth',0.85);
    end
    %Grafica los vecinos más cercanos
    line(X_train(1,n),X_train(2,n),'color',[.5 .5 .5],'marker','o',...
        'linestyle','none','markersize',15); %It appears that knnsearch has found
                                             % only the nearest eight neighbors. In fact, 
                                             % this particular dataset contains duplicate values
    % Define the center and diameter of a circle, based on the 
    % location of the new point y los grafica:
    if d(end)~=0
        ctr = X - d(end);
        diameter = 2*d(end);
        % Draw a circle around the k nearest neighbors:
        h = rectangle('position',[ctr',diameter,diameter], 'curvature',[1 1]);
        set(h,'linestyle',':')
    end
    

end
hold on; xlabel('Rombo= Train ; Asterisco=Val;  Círculo Color=Clasificación; Círculo gris= Vecinos: puntos=radio vecindad ' );

title(strcat('k-vecinos en IrisDataset, k=',num2str(k),', distancia="',distancia,'"'));
figure;plotconfusion(class_vtrue,class_val);

