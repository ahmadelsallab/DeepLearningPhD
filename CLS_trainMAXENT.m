% Function:
% Trains the MaxEnt model
% Inputs:
% mTrainFeatures: Input data. Matrix (nxm), where n is the number of examples and m is the features vector length
% mTrainTargets: Associated targets. Vector (nxl), where n is the number of examples and l is the number of target classes
% Output:
% MAXENT_clsParams: Parameters of the maxent model
function [MAXENT_clsParams] = CLS_trainMAXENT(mTrainFeatures, mTrainTargets)

    nNumTargets = size(mTrainTargets, 2);
    nNumFeatures = size(mTrainFeatures, 2);
    
    % Create the empty class model
    C0 = maxent(nNumTargets, nNumFeatures);
    
    % Get the train targets vector
    [I vTrainTargets]=max(mTrainTargets, [], 2);
    
    % Train the model
    MAXENT_clsParams = train(C0,vTrainTargets,mTrainFeatures,'none');
    
end % end function