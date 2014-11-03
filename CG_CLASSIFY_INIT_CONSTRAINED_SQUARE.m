% Function:
% Makes error back propagation for the top layer only of the NN, applying
% SQUARE penalty barrier method
% Inputs:
% VV: Vector of all class weights serialized row-wise
% Dim: Vector of sizes of top layer (input and output)
% wTopProbs: The activations at the input of the top layer
% target: The associated target
% wTopProbs_prev: The top layer activations of the previous mapping phase
% w_class_prev: The class weights of previous mapping phase
% alpha: Parameter of the SQUARE barrier penalty method 
% Output:
% f: The negative of the error
% df: The back-propagated delta (to be multiplied by input data to update
% the weigths
function [f, df] = CG_CLASSIFY_INIT_CONSTRAINED_SQUARE(VV, Dim, wTopProbs, target, wTopProbs_prev, w_class_prev, alpha);
	l1 = Dim(1);
	l2 = Dim(2);
	N = size(wTopProbs,1);

	% e_new
	w_class = reshape(VV,l1+1,l2);
	wTopProbs = [wTopProbs  ones(N,1)];  
	targetout = exp(wTopProbs*w_class);
	targetout = targetout./repmat(sum(targetout,2),1,size(target,2));
	f = -sum(sum( target(:,1:end).*log(targetout))) ;
	  
	%e_old
	wTopProbs_prev = [wTopProbs_prev  ones(N,1)]; 
	targetout_prev = exp(wTopProbs_prev*w_class_prev);
	targetout_prev = targetout_prev./repmat(sum(targetout_prev,2),1,size(target,2));
	e = -sum(sum( target(:,1:end).*log(targetout_prev))) ;
	  
	%g = min(0, (f-e))
	C = 0; % Reqularization term to move things if f = e
	g = min(0, (f-(e-C)));
	  
	% f = f + alpha*g^2
	f = f + alpha * g^2;
  
	%df = df/dw + 2*alpha*g*dg/dw = df/dw + 2*alpha*g*df/dw = df/dw*(1+2*alpha*g)
	% note: df/dw = wTopProbs'(targetout-target(:,1:end))
	IO = (targetout-target(:,1:end))*(1 + 2*alpha*g);
	Ix_class=IO; 
	dw_class =  wTopProbs'*Ix_class; 

	df = [dw_class(:)']'; 

