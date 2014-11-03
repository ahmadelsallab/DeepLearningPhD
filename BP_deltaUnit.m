% Function:
% Calculates the delta parameter in backprop by back propagating the delta error given at the top layer 
% to the input layer using the given network weights and their activations
% Inputs:
% NW_unitWeights: Matrix cell array {k}nxm, where k is the number of layers, n is
% the number of input neurons and m is the number of output neurons for
% each layer
% NW_unitWProbs: Cell array of each layer activation, augmented with ones at the last colomn
% Ix_upper: Vector of the delta error at the top layer of the unit
% Output:
% Ix: the delta of error propagated to the input layer of the unit


function [Ix] = BP_deltaUnit(NW_unitWeights, NW_unitWProbs, Ix_upper, w_upper)
	% delta_J
	Ix_up = Ix_upper;
	
	%numUnitLayers = size(NW_unitWeights, 2);
	layer = size(NW_unitWeights, 2);
	
	% delta_I = V (layer weights) * delta_J (upper delta) * f'(input data to layer) [f'=ip data * (1-ip data) for unipolar]
	% ex: Ix2 = (Ix3*w3').*w2probs.*(1-w2probs); 
	while(layer >= 1)
		Ix_up = (Ix_up * (w_upper)') .* NW_unitWProbs{layer} .* (1 - NW_unitWProbs{layer});
        Ix_up = Ix_up(:,1:end-1);
		w_upper = NW_unitWeights{layer};
        layer = layer - 1;
	end
	
	Ix = Ix_up;
end

