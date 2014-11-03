%load input_data;

% Get train targets as numbers
[I vTrainTargets]=max(mTrainTargets, [], 2);
vNumTrainExamplesPerClass = zeros(size(mTrainTargets, 2), 1);

%mTrainFeaturesByClass = zeros(size(mTrainTargets, 2), size(mTrainFeatures, 1), size(mTrainFeatures, 2));

mAverageExamplePerClass = zeros(size(mTrainTargets, 2), size(mTrainFeatures, 2));

% Split train features by target
cTrainFeaturesByClass = [];
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

% Classify train targets
vTargetOut = zeros(size(vTrainTargets, 1), size(vTrainTargets, 2));
for m = 1 : size(mTrainFeatures, 1)
    % Search for min dist.
    nMinDist = 1000000000;
    for k = 1 : size(mAverageExamplePerClass, 1)
        nDist = sum((mTrainFeatures (m, :) - mAverageExamplePerClass(k, :)).^2);
        if(nDist < nMinDist)
            nMinDist = nDist;
            vTargetOut(m) = k;
        end
    end
end

%[mConfusionMatrix, mNormalConfusionMatrix, vNumTrainExamplesPerClass, vAccuracyPerClass, nOverallAccuracy] = LM_buildConfusionMatrix(vTrainTargets, vTargetOut)

