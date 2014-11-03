% Function:
% Makes error back propagation for the NN for only top N layers. N is
% specified in configuration CONFIG_setConfigParams.
% Inputs:
% VV: Vector of all class weights serialized row-wise
% Dim: Vector of sizes of top layer (input and output)
% XX: Vector of the input data to the NN taken row-wise
% wTopProbs: The activations at the input of the top layer
% target: The associated target
% eMappingMode: See CONFIG_setConfigParams
% NW_unitWeights: Cell array of weights of each constituting unit of the net
% NW_weights_in: Weights of each layer
% w_class_in: Weights of the top class layer
% Output:
% f: The negative of the error
% df: The back-propagated delta (to be multiplied by input data to update
% the weigths
function [f, df] = CG_CLASSIFY_N_Layers(VV, Dim, wInputProbs, target, eMappingMode, NW_unitWeights, NW_weights_in, w_class_in);
l = Dim';
N = size(wInputProbs,1);
if (~strcmp(eMappingMode, 'DEPTH_BASE_UNIT_MAPPING'))
	% Do decomversion.
	N_layers = size(Dim, 1) - 2; % remove input and target layers
	offset = 0;
	for layer = 1 : N_layers
		NW_weights{layer} = reshape(VV((offset+1) : (offset+(l(layer)+1)*l(layer+1))), l(layer)+1, l(layer+1));
		offset = offset + (l(layer)+1)*l(layer+1);
	end
	w_class = reshape(VV(offset+1:offset+(l(N_layers+1)+1)*l(N_layers+2)), l(N_layers+1)+1, l(N_layers+2));
else
	N_layers = size(NW_weights_in, 2);
	NW_weights = NW_weights_in;
	w_class = w_class_in;
end
BP_layerInputData = wInputProbs;
if (~strcmp(eMappingMode, 'DEPTH_BASE_UNIT_MAPPING'))
	[activationTemp BP_wprobs] = NM_neuralNetActivation(BP_layerInputData, NW_weights);
else
	
	[activationTemp BP_wprobs unitInternalActivations NW_unitWProbs] = NM_compositeNetActivation(BP_layerInputData, NW_unitWeights);
end

targetout = exp(BP_wprobs{N_layers}*w_class);
targetout = targetout./repmat(sum(targetout,2),1,size(target,2));
f = -sum(sum( target(:,1:end).*log(targetout))) ;

IO = (targetout-target(:,1:end));
Ix_class=IO; 
dw_class =  (BP_wprobs{N_layers})'*Ix_class; 
 
layer = N_layers;
Ix_upper = Ix_class;
w_upper = w_class;
%baseUnit = 0; % at top level there's intermediate weight not base unit
while (layer >= 1)
	
	% delta_k = Ix{layer}
	% delta_j = delta_k * wJk' * f'(yink) = (Ix_upper*w_upper').*BP_wprobs{layer}.*(1-BP_wprobs{layer})
	%if (baseUnit == 0)
	if (~strcmp(eMappingMode, 'DEPTH_BASE_UNIT_MAPPING'))
		Ix{layer} = (Ix_upper*w_upper').*BP_wprobs{layer}.*(1-BP_wprobs{layer});
	else
		Ix{layer} = BP_deltaUnit(NW_unitWeights, NW_unitWProbs, Ix_upper);
	end
	Ix{layer} = Ix{layer}(:,1:end-1);
	if(layer ~= 1)
		dw{layer} = (BP_wprobs{layer-1})'*Ix{layer};
	else
		dw{layer} = XX'*Ix{layer};
    end
    Ix_upper = [];
	Ix_upper = Ix{layer};
	w_upper = [];
	w_upper = NW_weights{layer}; % in case of "base unit", NW_weights are the intermediate weights
        layer = layer - 1;
	% if(depthBaseUnitMapping ~= 0)
		% baseUnit = ~baseUnit; %switch to intermediate weight or base unit
	% end
	
end

df = [];
for(layer = 1 : N_layers)
	df = [df dw{layer}(:)'];
end
df = [df dw_class(:)'];
df = df';
