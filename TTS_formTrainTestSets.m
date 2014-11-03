% Function:
% Forms the train and test sets out of the raw data given. The criteria of
% split is defined in the inputs
% Inputs:
% mFeatures: Raw features. Matrix (nxm), where n is the number of examples and m is the features vector length
% mTargets: Raw targets. Matrix (nxl), where n is the number of examples and l is the number of target classes
% sCriteria: The splitting criteria
% nTrainToTestFactor: The ratio of splitting train to test sets
% Output:
% mTestFeatures: Test features. Matrix (nxm), where n is the number of examples and m is the features vector length
% mTestTargets: Test targetss. Matrix (nxl), where n is the number of examples and l is the number of target classes
% mTrainFeatures: Train features. Matrix (nxm), where n is the number of examples and m is the features vector length
% mTrainTargets: Train targets. Matrix (nxl), where n is the number of examples and l is the number of target classes
function [mTestFeatures, mTestTargets, mTrainFeatures, mTrainTargets] = TTS_formTrainTestSets(mFeatures, mTargets, sCriteria, nTrainToTestFactor)
    
    % Initialize the matrices
    mTestTargets = [];
    mTestFeatures = [];
    mTrainTargets = [];
    mTrainFeatures = [];
     
    % Randomize if criteria is random
    if strcmp(sCriteria, 'random')
        rand('state',0); %so we know the permutation of the training data
        randomorder=randperm(size(mFeatures,1));       
    end % end if

    for b = 1 : size(mFeatures,1)
        fprintf(1, 'Splitting example %d\n', b);
        % Read the raw features and targets according to split criteria
        if strcmp(sCriteria, 'random')
            mLocFeatures = mFeatures(randomorder(b), :);
            mLocTargets = mTargets(randomorder(b), :);
        elseif strcmp(sCriteria, 'uniform')
            mLocFeatures = mFeatures(b, :);
            mLocTargets = mTargets(b, :);
        end % end if-elseif
        
        % Feed test and train sets according to the ratio configured
        if (mod(b-1, nTrainToTestFactor) == 0)	
            mTestTargets = [mTestTargets; mLocTargets];
            mTestFeatures = [mTestFeatures; mLocFeatures];
        else
            mTrainTargets = [mTrainTargets; mLocTargets];
            mTrainFeatures = [mTrainFeatures; mLocFeatures];
        end; % end if-else
    end; % end for

end % end function