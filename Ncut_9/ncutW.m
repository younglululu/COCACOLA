function [NcutDiscrete,NcutEigenvectors,NcutEigenvalues,NcutEigenvectors1,EigenVectors2] = ncutW(W,nbcluster);
% [NcutDiscrete,NcutEigenvectors,NcutEigenvalues] = ncutW(W,nbcluster);
% 
% Calls ncut to compute NcutEigenvectors and NcutEigenvalues of W with nbcluster clusters
% Then calls discretisation to discretize the NcutEigenvectors into NcutDiscrete
% Timothee Cour, Stella Yu, Jianbo Shi, 2004

% compute continuous Ncut eigenvectors
[NcutEigenvectors,NcutEigenvalues] = ncut(W,nbcluster);

NcutEigenvectors1 = NcutEigenvectors;
% compute discretize Ncut vectors
[NcutDiscrete,NcutEigenvectors,EigenVectors2] =discretisation(NcutEigenvectors);

NcutDiscrete = full(NcutDiscrete);