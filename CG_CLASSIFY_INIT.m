% Function:
% Makes error back propagation for the top layer only of the NN
% Inputs:
% VV: Vector of all class weights serialized row-wise
% Dim: Vector of sizes of top layer (input and output)
% wTopProbs: The activations at the input of the top layer
% target: The associated target
% Output:
% f: The negative of the error
% df: The back-propagated delta (to be multiplied by input data to update
% the weigths
function [f, df] = CG_CLASSIFY_INIT(VV,Dim,wTopProbs,target);
l1 = Dim(1);
l2 = Dim(2);
N = size(wTopProbs,1);
% Do decomversion.
w_class = reshape(VV,l1+1,l2);
wTopProbs = [wTopProbs  ones(N,1)];  

targetout = exp(wTopProbs*w_class);
targetout = targetout./repmat(sum(targetout,2),1,size(target,2));
f = -sum(sum( target(:,1:end).*log(targetout))) ;
IO = (targetout-target(:,1:end));
Ix_class=IO; 
dw_class =  wTopProbs'*Ix_class; 

df = [dw_class(:)']'; 

