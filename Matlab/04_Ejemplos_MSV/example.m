clc; clearvars ; close all;

%Load Fisher's iris data set. 
% load fisheriris
meas = csvread('../Dataset/Train_Input.csv');
species = readcell('../Dataset/Train_TargetsName.csv');
%Remove the sepal lengths and widths, and all observed setosa irises.
