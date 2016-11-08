function [SegLabel]= NcutImage(W,nbSegments)
%  [SegLabel,NcutDiscrete,NcutEigenvectors,NcutEigenvalues,W,imageEdges]= NcutImage(I);
%  Input: I = brightness image
%         nbSegments = number of segmentation desired
%  Output: SegLable = label map of the segmented image
%          NcutDiscrete = Discretized Ncut vectors
%  
% Timothee Cour, Stella Yu, Jianbo Shi, 2004.


 
if nargin <2,
   nbSegments = 10;
end

%[W,imageEdges] = ICgraph(I);

[SegLabel,~,~] = ncutW(W,nbSegments);
