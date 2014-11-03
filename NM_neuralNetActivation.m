% Function:
% Gives the activation of each network/classifier layer, both in augmented and raw form
% Inputs:
% weight: Matrix cell array {k}nxm, where k is the number of layers, n is
% the number of input neurons and m is the number of output neurons for
% each layer
% data: Matrix nxm, where n is the number of examples and m is the lenght of each example. data is not augmented
% Output:
% activation: Cell array of layers activations without ones colomn augmentation
% augmentedActivation: same as activation with ones colomn augmentation

function [activation augmentedActivation] = NM_neuralNetActivation(data, weights)
  data = [data ones(size(data, 1) ,1)];
  layerInputData = data;

  for(layer = 1 : size(weights, 2))
		
    [layerActivation augmentedLayerActivation]= NM_layerActivation(layerInputData, weights{layer});

    activation{layer} = layerActivation;
    augmentedActivation{layer} = augmentedLayerActivation;
    
    layerInputData = [];
    layerInputData = augmentedLayerActivation;
    
  end

end