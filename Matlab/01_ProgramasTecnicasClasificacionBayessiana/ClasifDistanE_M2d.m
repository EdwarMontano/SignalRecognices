% Clasificación Mínima Distancia Euclidiana y Mahalanobis
% The optimal Bayesian classifier is significantly simplified under the following assumptions:
% • The classes are equiprobable.
% • The data in all classes follow Gaussian distributions.
% • The covariance matrix is the same for all classes.
% • The covariancematrix is diagonal and all elements across the diagonal are equal.That is, S = ?2I,
% where I is the identity matrix.

clc; clearvars; close all;

%% Data Generation for ilustration

% Parameters:
mu1=[0 ; 0];   mu2=[3 ; 3]; % promedios
Pw1=0.5; Pw2=0.5; %Probabilidad A Priori de Clases


SIG1=[1.1  0.3; 0.3  1.9];
SIG2=[1.1  0.3; 0.3  1.9];

% SIG1=[0.3 0.0; 0.0  0.35];
% SIG2=[1.1  0.3; 0.3  1.9];


% SIG1=[1.1  0.3; 0.3  1.9];
% SIG2=[0.3 0.1; 0.1 0.35];

% SIG1=[1.2  0.4; 0.4 1.8];
% SIG2=[1.5  0.5; 0.5  1.2];

% SIG1=[1.0  0.0; 0.0 1.0];
% SIG2=[1.5  0.0; 0.0  1.2];

% Generar Datos Clase 1 (d=2)
ND1=100; % Número vectores 
dat1 = mvnrnd(mu1,SIG1,ND1); %data set Clase 1

% Generar Datos Clase 2 (d=2)
ND2=100; % Número vectores
dat2 = mvnrnd(mu2,SIG2,ND2); %data set Clase 2

%Visualización
plot(dat1(:,1), dat1(:,2),'b+',dat2(:,1), dat2(:,2),'rx')
xlabel('x1'); ylabel('x2'); title('MINIMUN DISTANCE CLASSIFIERS')
hold on;
plot(mu1(1),mu1(2),'bs' ,'MarkerSize',10,  'LineWidth',2) 
plot(mu2(1),mu2(2),'rs' ,'MarkerSize',10,  'LineWidth',2)


%% vector to classify 
%x=[1.0; 2.2];  
x=[2.5 ; 0.0];
% %x=[1 ; 3.1];
% x=[1.5 ; 2.0];
%x=[1 ; -0.5];
% x=[3; -0.5];

hold on; plot(x(1),x(2), 'dk');

%% Clasification: Euclidian Distance
de1 = sqrt((x-mu1)'*(x-mu1))
de2 = sqrt((x-mu2)'*(x-mu2))
if de1<de2
    display('Distancia Euclidiana: x se asigna a la clase w1');
    line([mu1(1)  x(1)],[mu1(2)  x(2)],'Color','m','LineStyle','--')
else
    display('Distancia Euclidiana: x se asigna a la clase w2');
    line([mu2(1)  x(1)],[mu2(2)  x(2)],'Color','m','LineStyle','--')
end

%% Clasification: Mahalanobis Distance
dm1 = sqrt((x-mu1)'*inv(SIG1)*(x-mu1))
dm2 = sqrt((x-mu2)'*inv(SIG2)*(x-mu2))
if dm1<dm2
    display('Distancia Mahalanobis: x se asigna a la clase w1');
    line([mu1(1)  x(1)],[mu1(2)  x(2)],'Color','g','LineStyle','--')
else
    display('Distancia Mahalanobis: x se asigna a la clase w2');
    line([mu2(1)  x(1)],[mu2(2)  x(2)],'Color','g','LineStyle','--')
end
                                                                                                                                    
% Display Legend 
legend('Class1','Class2','Mean1','Mean2','New vector', 'D. Eucl.', 'D. Mahal')

Msg1=['x-->w1=' num2str(de1)]; Msg2=['x-->w2=' num2str(de2)];
Msg3=['x-->w1=' num2str(dm1)]; Msg4=['x-->w2=' num2str(dm2)];


msgbox({'Euclidiana';Msg1;Msg2;'Mahalanobis';Msg3;Msg4},'Clasificador Distancia');

