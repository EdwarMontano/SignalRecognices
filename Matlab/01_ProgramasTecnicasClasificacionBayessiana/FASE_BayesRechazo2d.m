clear all; close all; clc;

Xtrain = csvread('../Dataset/Train_Input.csv');
ytrain = csvread('../Dataset/Train_Targets.csv');
Xtest = csvread('../Dataset/Test_Input.csv');
ytest = csvread('../Dataset/Test_Targets.csv');
data1=Xtrain(101:200,2:3);
datab=Xtrain(201:300,2);
datag=Xtrain(201:300,3);
data2=[datab datag]
meanData1=mean(data1);
meanData2=mean(data2);
stdData1=cov(data1);
stdData2=cov(data2);
% Datos conocidos sobre las distribuciones de probabilidad de clases
num_clases =2; m1=meanData1'; m2 = meanData1'; S1=stdData1;
% S1=eye(2); 
S2=stdData2;
p1 = 1/3; p2=1/3;


% si se cambian las probabilidades a p1 = 0.3; p2=0.7; cambia la respuesta
% de pertenece a la clase 1 a pertenece a la clase 2
% evaluar las implicaciones de: S1=0.2*eye(2); S1=2*eye(2); 
% S1=[1 0.3; 0.2 1];

% datos nuevos que se desean clasificar
xp=[-0.824328300105256	-2.13332514386491]';
% xp=[1.46002561914480	-0.377491893215636]'; 
% xp=[-1.25147822126397	1.80923597630479]';

% Se verifica la probabilidad de que el dato en x1 pertenezca a la clase 1
constante1= (1/(((2*pi)^(num_clases/2))*(sqrt(det(S1)))));
pg1 = constante1*exp((-0.5)*(xp-m1)'*inv(S1)*(xp-m1));
pc1= p1*pg1

% Se verifica la probabilidad de que el dato en x1 pertenezca a la clase 2
constante2= (1/(((2*pi)^(num_clases/2))*(sqrt(det(S2)))));
pg2 = constante2*exp((-0.5)*(xp-m2)'*inv(S2)*(xp-m2));
pc2= p2*pg2

% Toma de decisi�n
if pc1>pc2
    disp('pertenece a la clase 2');
else
    disp('pertenece a la clase 3');
end

% Se verifica la probabilidad de clases en un conjunto de puntos y1
[x,y]=meshgrid(-5:0.1:6,-5:0.1:6);
[n,m]=size(x);
for i=1:n
    for j=1:m
        xp_g = [x(i,j) y(i,j)]';
        py1(i,j) = p1*constante1*exp((-0.5)*(xp_g-m1)'*inv(S1)*(xp_g-m1));
        py2(i,j) = p2*constante2*exp((-0.5)*(xp_g-m2)'*inv(S2)*(xp_g-m2));
    end
end
plot3(x,y,py1,'g');
grid on;
hold on;
plot3(x,y,py2,'b');
plot3(xp(1),xp(2),pc1,'rx');
plot3(xp(1),xp(2),pc2,'rx');
line([xp(1) xp(1)],[xp(2) xp(2)],[max(pc1,pc2) 0],'LineStyle','--','Color',[1 0 0])

