% Function:
% Trains the MaxEnt model
% Inputs:
% mTrainFeatures: Input data. Matrix (nxm), where n is the number of examples and m is the features vector length
% mTrainTargets: Associated targets. Vector (nxl), where n is the number of examples and l is the number of target classes
% Output:
% mAverageExamplePerClass: The average features vector per class
function [mAverageExamplePerClass] = CLS_trainAVG(mTrainFeatures, mTrainTargets)

% Get train targets as numbers
    [I vTrainTargets]=max(mTrainTargets, [], 2);
    vNumTrainExamplesPerClass = zeros(size(mTrainTargets, 2), 1);

    mAverageExamplePerClass = zeros(size(mTrainTargets, 2), size(mTrainFeatures, 2));
    
    for i = 1 : size(vTrainTargets, 1)
        vNumTrainExamplesPerClass(vTrainTargets(i)) = vNumTrainExamplesPerClass(vTrainTargets(i)) + 1;
        %TrainFeaturesByClass(vTrainTargets(i), vNumTrainExamplesPerClass(vTrainTargets(i)), :) = mTrainFeatures(i,:);

        % Accunulate features
        mAverageExamplePerClass(vTrainTargets(i),:) = mAverageExamplePerClass(vTrainTargets(i),:) + mTrainFeatures(i,:);
    end

    % Average
    for j = 1 : size(vNumTrainExamplesPerClass, 1)
        mAverageExamplePerClass(j, :) = mAverageExamplePerClass(j, :) ./ vNumTrainExamplesPerClass(j);
    end
    
end % end function