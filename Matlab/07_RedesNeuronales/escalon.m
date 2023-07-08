function out=escalon(in) 
% ESCALON.M 
% Programa que implementa una función de activación 
% tipo escalón, invocada desde cuatro.m 
for i=1:4 
    if in(i)>=0 
        out(i)=1; 
    else
        out(i)=0; % Si bipolar out(i) = -1;
    end; 
end; 
