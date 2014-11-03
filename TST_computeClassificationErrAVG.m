% Function:
% Report accuracy and error of Average Classifier
% Inputs:
% mTrainTargets, mTestTargets: Associated targets. Vector (nxl), where n is the number of examples and l is the number of target classes
% mAverageExamplePerClass: Mean class example
% Output:
% nTestErr, nTrainErr: number of misclassified examples
% vTrainTargetOut, vTestTargetOut: The vector targets of train and test
% sets
function [nTrainErr, nTestErr, vTrainTargetOut, vTestTargetOut] = TST_computeClassificationErrAVG(mTrainFeatures, mTrainTargets, mTestFeatures, mTestTargets, mAverageExamplePerClass)
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%% TRAIN ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Classify train targets
    vTrainTargetOut = zeros(size(vTrainTargets, 1), size(vTrainTargets, 2));
    for m = 1 : size(mTrainFeatures, 1)
        % Search for min dist.
        nMinDist = 1000000000;
        for k = 1 : size(mAverageExamplePerClass, 1)
            nDist = sum((mTrainFeatures (m, :) - mAverageExamplePerClass(k, :)).^2);
            if(nDist < nMinDist)
                nMinDist = nDist;
                vTrainTargetOut(m) = k;
            end
        end
    end

%%%%%%%%%%%%%%%%%%%%%%%%%%%% TEST ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Classify test targets
    vTestTargetOut = zeros(size(vTestTargets, 1), size(vTestTargets, 2));
    for m = 1 : size(mTestFeatures, 1)
        % Search for min dist.
        nMinDist = 1000000000;
        for k = 1 : size(mAverageExamplePerClass, 1)
            nDist = sum((mTestFeatures (m, :) - mAverageExamplePerClass(k, :)).^2);
            if(nDist < nMinDist)
                nMinDist = nDist;
                vTestTargetOut(m) = k;
            end
        end
    end
end % end function