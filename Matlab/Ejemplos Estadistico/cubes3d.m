clc; clearvars; close all ; %close all hidden 


%Cube 1
LadosC1= [5 9 5];
OrigC1= [ 2 2 6];
plotcube(LadosC1,OrigC1,.4,[1 0 0]); 
%Cube 2
LadosC2= [9  5  8];
OrigC2= [ 10 10 10];
plotcube(LadosC2,OrigC2,.4,[0 1 0]); 
%Cube 3
LadosC3= [5  5  5];
OrigC3= [ 18 18 18];
plotcube(LadosC3,OrigC3,.4,[0  0 1]); 

%
xlabel('x1');ylabel('x2');zlabel('x3'); 
axis([0  25  0  25  0  25]);
grid on;

% Draw k points in the cube
hold on;
k=9;

%Cube 1
u1 = OrigC1+LadosC1/2;
S1 = min(LadosC1)/4*eye(3);
x1 = mvnrnd(u1,S1,k)
plot3(x1(:,1),x1(:,2),x1(:,3) , 'o k')
%Cube 2
u2 = OrigC2+LadosC2/2;
S2 = min(LadosC2)/4*eye(3);
x2 = mvnrnd(u2,S2,k)
plot3(x2(:,1),x2(:,2),x2(:,3) , 'o k')
%Cube 3
u3 = OrigC3+LadosC3/2;
S3 = min(LadosC3)/4*eye(3);
x3 = mvnrnd(u3,S3,k)
plot3(x3(:,1),x3(:,2),x3(:,3) , 'o k')
