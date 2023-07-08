clear all; close all; clc;

load fisheriris;

n = size(meas,1);
rng(1) % For reproducibility
idxTrn = false(n,1);
idxTrn(randsample(n,round(0.5*n))) = true; % Training set logical indices
idxVal = idxTrn == false;                  % Validation set logical indices

tree = fitctree(meas(idxTrn,:),species(idxTrn));
view(tree,'Mode','Graph');
Ynew = predict(tree,meas(idxVal,:));
error = sum(~strcmp(Ynew,species(idxVal)));
disp(strcat('Porcentaje de error validación =',num2str(error)));
