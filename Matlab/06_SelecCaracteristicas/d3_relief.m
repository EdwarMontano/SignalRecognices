clear all; clc;

meas = csvread('../Dataset/Train_Input.csv');
species = readcell('../Dataset/Train_TargetsName.csv');
k=1;
relieff(meas,species,k) 