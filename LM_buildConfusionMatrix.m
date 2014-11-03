function [mConfusionMatrix, mNormalConfusionMatrix, vNumTrainExamplesPerClass, vAccuracyPerClass, nOverallAccuracy] = LM_buildConfusionMatrix(vDesiredTargets, vObtainedTargets)

    % Size of matrix is the maximum ID of classes
    nSizeOfMatrix = max(vDesiredTargets);
    mConfusionMatrix = zeros(nSizeOfMatrix, nSizeOfMatrix);
    mNormalConfusionMatrix = zeros(nSizeOfMatrix, nSizeOfMatrix);
    vNumTrainExamplesPerClass = zeros(nSizeOfMatrix, 1);
    for i = 1 : size(vDesiredTargets, 1)
        vNumTrainExamplesPerClass(vDesiredTargets(i)) = vNumTrainExamplesPerClass(vDesiredTargets(i)) + 1;
        %for j = 1 : size(vObtainedTargets, 2)
           mConfusionMatrix(vDesiredTargets(i),vObtainedTargets(i)) = mConfusionMatrix(vDesiredTargets(i),vObtainedTargets(i)) + 1;
        %end
    end
    
    % Normalize the confusion matrix per colomn
%     for m = 1 : size(mConfusionMatrix, 2)
%         if(sum(mConfusionMatrix(:, m)) ~= 0)
%             mNormalConfusionMatrix(:, m) = mConfusionMatrix(:, m)./sum(mConfusionMatrix(:, m));
%         else
%             mNormalConfusionMatrix(:, m) = rand(nSizeOfMatrix, 1);
%         end
%     end

    % Normalize colomn-wise
%     for m = 1 : size(mConfusionMatrix, 2)
%         if(sum(mConfusionMatrix(:, m)) ~= 0)
%             mNormalConfusionMatrix(:, m) = mConfusionMatrix(:, m)./sum(mConfusionMatrix(:, m));
%         else
%             mNormalConfusionMatrix(:, m) = 0;
%         end
%     end

    for m = 1 : size(mConfusionMatrix, 1)
        mNormalConfusionMatrix(m, :) = mConfusionMatrix(m, :)./sum(mConfusionMatrix(m, :));
    end
    
    vAccuracyPerClass = zeros(nSizeOfMatrix, 1);
    vCorrectlyClassifiedExamplesPerClass = diag(mConfusionMatrix);
    for n = 1 : size(vNumTrainExamplesPerClass, 1)
        vAccuracyPerClass(n) = vCorrectlyClassifiedExamplesPerClass(n) / vNumTrainExamplesPerClass(n);
    end
    
    nOverallAccuracy = sum(diag(mConfusionMatrix))/sum(vNumTrainExamplesPerClass);

end