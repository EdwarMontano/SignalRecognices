function [px]=estimar_kNN_density(Xtest,Xlear, knn)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FUNCTION
%   [px]=estimar_kNN_density(Xtest,Xlear,knn)
% k-nn-based approximation of a pdf. 
% The Distance Euclidian Distance is used for the volume calcul
%
% INPUT ARGUMENTS:
%   Xtest:          dxNt matrix test with Nt vectors
%   Xlear;          dxNl matrix learning with Nl vectors
%   knn:            number of nearest neghibors.
%
% OUTPUT ARGUMENTS:
%   px:             a vector, each element of which contains the estimate
%                   of p(x) for each vector of Xtest 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[dt,Nt]=size(Xtest);
[d,Nl]=size(Xlear);

if dt ~= d 
    px=[];
    fprintf('Feature set has more than one dimensions ');
    return;
end    

for  i=1: Nt  %Evaluar cada vector del data set de test 
    
    eucl=[];
    for j=1:Nl  % Calcular las distancias entre el vector de test y los del data set de aprendizaje 
        eucl(j)=sqrt(sum((Xtest(:,i)-Xlear(:,j)).^2));
    end
    eucl=sort(eucl,'ascend'); 
    ro=eucl(knn); %Asignar radio igual a la distancia del vector de la posición knn
    V= pi^(d/2) *ro^d /gamma(d/2+1); 
    px(i)=knn/(Nl*V);
end

