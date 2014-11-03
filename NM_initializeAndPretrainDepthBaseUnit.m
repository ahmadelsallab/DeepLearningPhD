% Function:
% Initializes the mapped unit weights with the basic unit weights, in addition to adjustment of the intermediate weights between the current unit and the lower.
% Inputs:
% ctrUnit: The index of the unit to be initialized
% NM_strNetParams: The network parameters
% Output:
% NM_strNetParams: The updated network parameters
function [NM_strNetParams] = NM_initializeAndPretrainDepthBaseUnit(NM_strNetParams, ctrUnit)
    
    % Replicate base unit weights
    NM_strNetParams.cUnitWeights{ctrUnit} = NM_strNetParams.cBaseUnitWeights;
	
    % Unless the fist unit (basic one); then modify the first layer. For
    % the basic unit, the first layer visible width = data width, for other
    % units (next ones), the first layer visible width = the width of the top of the previous
    % unit hidden width
	if ctrUnit ~= 1
        % The first layer (lower = visible) of next unit (unless the first) is the last one
        % of the previous unit (or the basic unit)
		nFirstLayerWidth = size(NM_strNetParams.cBaseUnitWeights{end}, 2) + 1;
        
        % The second (upper = hidden) is same as the basic unit first layer
        % hidden width
		nSecondLayerWidth = size(NM_strNetParams.cBaseUnitWeights{1}, 2);
		NM_strNetParams.cUnitWeights{ctrUnit}{1} = 0.1 * randn(nFirstLayerWidth, nSecondLayerWidth);
    end % end if
	
    % Initialize the intermediate weights linking the current unit to the
    % lower one
	NM_strNetParams.cWeights{ctrUnit} = NM_strNetParams.cUnitWeights{ctrUnit}{1};
    
end % end function
