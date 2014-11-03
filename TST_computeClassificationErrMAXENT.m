% Function:
% Report accuracy and error of MaxEnt model
% Inputs:
% mTrainTargets, mTestTargets: Associated targets. Vector (nxl), where n is the number of examples and l is the number of target classes
% MAXENT_clsParams: HMM transition and emission probs
% Output:
% nTestErr, nTrainErr: number of misclassified examples
% nTestAccuracy, nTrainAccuracy: percent of correctly classified examples
function [nTrainErr, nTestErr, nTestAccuracy, nTrainAccuracy] = TST_computeClassificationErrMAXENT(mTrainFeatures, mTrainTargets, mTestFeatures, mTestTargets, MAXENT_clsParams)
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%% TRAIN ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Get the train target vector
    [I vTrainTargets]=max(mTrainTargets, [], 2);

    % Get the train accuracy
    nTrainAccuracy = accuracy(MAXENT_clsParams, vTrainTargets, mTrainFeatures);

    % Obtain the output train targets
    vTrainTargetsOut = map(MAXENT_clsParams, mTrainFeatures);

    % Compute number of misclassified examples
    nTrainErr = size(find(vTrainTargets ~= vTrainTargetsOut), 1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%% TEST ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Get the test target vector
    [I vTestTargets]=max(mTestTargets, [], 2);
    
    % Get the train accuracy
     nTestAccuracy = accuracy(MAXENT_clsParams, vTestTargets, mTestFeatures);
     
     % Obtain the output test targets
     vTestTargetsOut = map(MAXENT_clsParams, mTestFeatures);
     
     % Compute number of misclassified examples
     nTestErr = size(find(vTestTargets ~= vTestTargetsOut), 1);

end % end function