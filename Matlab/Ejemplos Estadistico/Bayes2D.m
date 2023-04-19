clear all; close all; clc;

% Datos conocidos sobre las distribuciones de probabilidad de clases
num_clases =2; m1=1; m2 = 3;
S1=eye(1); 
S2=S1;
p1 = 0.5; p2=0.5; %Probailidades a priori

% S = [?1^2 ?12; ?12 ?2^2]
% When the two coordinates of x are uncorrelated (?12 = 0) and their variances are equal, the data
% vectors form “spherically shaped” clusters.
% When the two coordinates of x are uncorrelated (?12 = 0) and their variances are unequal, the data
% vectors form “ellipsoidally shaped” clusters.
% When the two coordinates of x are correlated (?12 = 0), themajor and minor axes of the ellipsoidally
% shaped cluster are no longer parallel to the axes. The degree of rotation with respect to the axes
% depends on the value of ?12

% si se cambian las probabilidades a p1 = 0.3; p2=0.7; cambia la respuesta
% de pertenece a la clase 1 a pertenece a la clase 2
% evaluar las implicaciones de: S1=0.2*eye(2); S1=2*eye(2); 
% S1=[1 0.3; 0.2 1];

% datos nuevos que se desean clasificar
xp=1.8;

% Se verifica la probabilidad de que el dato en x1 pertenezca a la clase 1
constante1= (1/(((2*pi)^(num_clases/2))*(sqrt(det(S1)))));
pg1 = constante1*exp((-0.5)*(xp-m1)'*inv(S1)*(xp-m1));
pc1= p1*pg1

% Se verifica la probabilidad de que el dato en x1 pertenezca a la clase 2
constante2= (1/(((2*pi)^(num_clases/2))*(sqrt(det(S2)))));
pg2 = constante2*exp((-0.5)*(xp-m2)'*inv(S2)*(xp-m2));
pc2= p2*pg2

% Toma de decisión
if pc1>pc2
    disp('pertenece a la clase 1');
else
    disp('pertenece a la clase 2');
end


% Se verifica la probabilidad de clases en un conjunto de puntos y1
y1=(0:0.1:6);
l=length(y1);
for i=1:l 
    
    py1(i) = p1*constante1*exp((-0.5)*(y1(i)-m1)'*inv(S1)*(y1(i)-m1));
    py2(i) = p2*constante2*exp((-0.5)*(y1(i)-m2)'*inv(S2)*(y1(i)-m2));
end
plot(y1,py1,'g');
grid on;
hold on;
plot(y1,py2,'b');
plot(xp,pc1,'bx');
plot(xp,pc2,'bx');
line([xp xp],[max(pc1,pc2) 0],'LineStyle','--','Color',[1 0 0])
