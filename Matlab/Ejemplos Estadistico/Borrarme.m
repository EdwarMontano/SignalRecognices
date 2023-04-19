load fisheriris
%Use the default Gaussian distribution and a confusion matrix:
O1 = fitNaiveBayes(meas,species);
C1 = O1.predict(meas);
cMat1 = confusionmat(species,C1) 