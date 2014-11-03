% Function:
% Gives the activation of single layer NN, both in augmented and raw form
% Inputs:
% weight: Matrix nxm, where n is the number of input neurons and m is the number of output neurons
% data: Matrix nxm, where n is the number of examples and m is the lenght of each example. data is not augmented
% Output:
% activation: output layer activation without ones colomn augmentation
% augmentedActivation: same as activation with ones colomn augmentation

function [activation, augmentedActivation] = NM_layerActivation(data, weight)
    activation = 1./(1 + exp(-data*weight));
    augmentedActivation = [activation ones(size(data,1), 1)];
end