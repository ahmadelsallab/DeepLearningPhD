% Function:
% Gives the activation of base network/classifier top unit, both in augmented and raw form
% Inputs:
% weight: Cell Array of cell arrays: {l}x{k}nxm, where l is the number of base unit NW's = number of layers composed of base units, 
% k is the number of layers in each base unit, n is
% the number of input neurons and m is the number of output neurons for
% each layer
% data: Matrix nxm, where n is the number of examples and m is the lenght of each example. data is not augmented
% Output:
% activation: Cell array of output unit activations without ones colomn augmentation FOR
% EACH UNIT OF BASE UNITS
% augmentedActivation: same as activation with ones colomn augmentation

function [activation augmentedActivation unitInternalActivations augUnitInternalActivations] = NM_compositeNetActivation(data, weights)

  %data = [data ones(size(data, 1) ,1)];
  baseUnitInputData = data;
  
  for(unit = 1 : size(weights, 2))
  
   [baseUnitActivation augmentedBaseUnitActivation] = NM_baseUnitActivation(baseUnitInputData, weights{unit});
   [unitInternalActivations{unit} augUnitInternalActivations{unit}] = NM_neuralNetActivation(baseUnitInputData, weights{unit});
    
   activation{unit} = baseUnitActivation;
   augmentedActivation{unit} = augmentedBaseUnitActivation;
	
   baseUnitInputData = [];
	baseUnitInputData = baseUnitActivation;
    
  end
  
end