% Maximum Likelihood Parameter Estimation of Gaussian pdfs

% One problem oftenmet in practice is that the pdfs describing the statistical distributionof the data in the
% classes are not known and must be estimated using the training data set.

clear all; close all; clc;

load('iris_dataset');

N =50;
N_train = 30;
num_clases = 3;
class = ones(1,150);class(1,51:100)=2;class(1,101:end)=3;

% Se usan los primeros N_train para entrenamiento, esto es, estimar la
% media y covarianza para cada clase usando sólo las características 3 y 4

m1=mean(irisInputs(3:4,1:N_train),2);
m2=mean(irisInputs(3:4,51:51+N_train),2);
m3=mean(irisInputs(3:4,101:101+N_train),2);
m=[m1 m2 m3];
S1=cov(irisInputs(3:4,1:N_train)');
S2=cov(irisInputs(3:4,51:51+N_train)');
S3=cov(irisInputs(3:4,101:101+N_train)');
p1=1/3;p2=1/3;p3=1/3;

% Gráfica de los puntos de entrenamiento
X1=[irisInputs(3,1:N_train) irisInputs(3,51:51+N_train) irisInputs(3,101:101+N_train)];
X2=[irisInputs(4,1:N_train) irisInputs(4,51:51+N_train) irisInputs(4,101:101+N_train)];
C = [class(1,1:N_train) class(1,51:51+N_train) class(1,101:101+N_train)];
scatter(X1,X2,30,C);
hold on;grid on;
%Grafica de las medias de cada clase
plot(m1(1),m1(2),'rx');
plot(m2(1),m2(2),'rx');
plot(m3(1),m3(2),'rx');

%Validación
X1_val=[irisInputs(3,N_train+1:50) irisInputs(3,51+N_train:100) irisInputs(3,101+N_train:end)];
X2_val=[irisInputs(4,N_train+1:50) irisInputs(4,51+N_train:100) irisInputs(4,101+N_train:end)];
L_val = (50-N_train);
class_val =zeros(3,L_val*3);
class_val(1,1:L_val)=1;
class_val(2,L_val+1:L_val*2)=1;
class_val(3,2*L_val+1:end)=1;
class_vtrue =zeros(3,L_val*3);

constante1= (1/(((2*pi)^(num_clases/2))*(sqrt(det(S1)))));
constante2= (1/(((2*pi)^(num_clases/2))*(sqrt(det(S2)))));
constante3= (1/(((2*pi)^(num_clases/2))*(sqrt(det(S3)))));

for i=1:L_val*3
    X = [X1_val(i);X2_val(i)];
    
    %Calcula las probabilidades de clase
    % Se verifica la probabilidad de que el dato en x1 pertenezca a la clase 1
    pc1 = p1*constante1*exp((-0.5)*(X-m1)'*inv(S1)*(X-m1));
    % Se verifica la probabilidad de que el dato en x1 pertenezca a la clase 2
    pc2 = p2*constante2*exp((-0.5)*(X-m2)'*inv(S2)*(X-m2));
    % Se verifica la probabilidad de que el dato en x1 pertenezca a la clase 3
    pc3 = p3*constante3*exp((-0.5)*(X-m3)'*inv(S3)*(X-m3));
    [d,j1]=max(class_val(:,i));
    % Se elige la clase con mayor probabilidad
    [d,j2]=max([pc1 pc2 pc3]);
    class_vtrue(j2,i)=1;
    %Grafica del punto de entrada con un color que depende de la clase
    switch j1
        case (1)
             plot(X(1),X(2),'k*');
        case (2)
             plot(X(1),X(2),'b*');
        otherwise
             plot(X(1),X(2),'g*');
    end
    %Grafica la clase con mayor probabilidad
    line([X(1) m(1,j2)],[X(2) m(2,j2)],'LineStyle','--','Color',[1 0 0]);
end
title('Máxima Verosimilitud en IrisDataset');
figure;plotconfusion(class_vtrue,class_val);

