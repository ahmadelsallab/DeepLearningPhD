% Function:
% Report accuracy and error of Average Classifier
% Inputs:
% mTrainTargets, mTestTargets: Associated targets. Vector (nxl), where n is the number of examples and l is the number of target classes
% TFIDF_clsParams: The TFIDF weights
% Output:
% nTestErr, nTrainErr: number of misclassified examples
% vTrainTargetOut, vTestTargetOut: The vector targets of train and test
% sets
function [nTrainErr, nTestErr, vTrainTargetsOut, vTestTargetsOut] = TST_computeClassificationErrTFIDF(mTrainFeatures, mTrainTargets, mTestFeatures, mTestTargets, TFIDF_clsParams)        
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%% TRAIN ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [I vTrainTargets]=max(mTrainTargets, [], 2);
    vTrainScores = TFIDF_clsParams.mWeightsTFIDF * mTrainFeatures';
    [X vTrainTargetsOut] = max(vTrainScores, [], 1);
    vTrainTargetsOut = vTrainTargetsOut';
    nTrainErr = size(find(vTrainTargets ~= vTrainTargetsOut), 1);
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%% TEST ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [I vTestTargets]=max(mTestTargets, [], 2);
    vTestScores = TFIDF_clsParams.mWeightsTFIDF * mTestFeatures';
    [X vTestTargetsOut] = max(vTestScores, [], 1);
    vTestTargetsOut = vTestTargetsOut';
    nTestErr = size(find(vTestTargets ~= vTestTargetsOut), 1);
        
end % end function