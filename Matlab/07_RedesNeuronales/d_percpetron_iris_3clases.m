clear all; clc; close all;

load fisheriris
xi=meas(:,2:4)';
x=mapstd(xi);
%mean(x1'),std(x1),
yd=zeros(2,150);
yd(1,51:end)=1;
yd(2,101:end)=1;
plotpv(x,yd);
grid on;

PS=[min(x(1,:)) max(x(1,:)); min(x(2,:)) max(x(2,:)); min(x(3,:)) max(x(3,:))];
red=newp(PS,2);
red.trainParam.epochs = 100;
red.trainParam.goal = 0.00009;

red=train(red,x,yd);
Pesos=red.iw{1,1};
Bias=red.b{1};
figure(1);hold on;
plotpc(Pesos,Bias)

y=sim(red,x);
error=abs(y-yd);
sum(sum(error))

% para la matriz de confusion se reconfigura la salida
ydc =zeros(3,150);
ydc(1,1:50)=1;
ydc(2,51:100)=1;
ydc(3,101:150)=1;
yc =zeros(3,150);

for n=1:150
    if  (y(:,n)==[0 0]')
        yc(:,n)=[0 0 0]';
    else
        if (y(:,n)==[1 0]')
            yc(:,n)=[0 1 0]';
        else
            if (y(:,n)==[1 1]')
                yc(:,n)=[0 0 1]';
            else
                yc(:,n)=[1 1 1]';
            end
        end
    end
end
        
figure;
plotconfusion(yc,ydc)