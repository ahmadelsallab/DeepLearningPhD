% Function:
% Converts the given 2-D matrix into 3-D one with random splits of size
% nBatchSize.
% Inputs:
% nBatchSize: The batch size in number of examples
% mFeatures: Test features. Matrix (nxm), where n is the number of examples and m is the features vector length
% mTargets: Test targets. Matrix (nxl), where n is the number of examples and l is the number of target classes
% Output:
% mBatchData, mBatchTargets: Matrices (nBatchSize X nNumFeatures X nNumBatches) similar to input ones
function [mBatchData, mBatchTargets] = BM_randomizeBatchData(nBatchSize, mFeatures, mTargets)
    
    % Get number of train and test examples
    nNumExamples = size(mFeatures, 1);
    
    % Get number of targets
    nNumTargets = size(mTargets, 2);
    
    % Get number of features
    nNumFeatures  =  size(mFeatures,2);
    
    rand('state',0); %so we know the permutation of the training data
    randomorder=randperm(nNumExamples);

    nNumBatches=floor(nNumExamples/nBatchSize);
    mBatchData = zeros(nBatchSize, nNumFeatures, nNumBatches);
    mBatchTargets = zeros(nBatchSize, nNumTargets, nNumBatches);

    for b=1:nNumBatches
	  fprintf(1, 'Making Batch %d out of %d\n', b, nNumBatches);
      mBatchData(:,:,b) = mFeatures(randomorder(1+(b-1)*nBatchSize:b*nBatchSize), :);
      mBatchTargets(:,:,b) = mTargets(randomorder(1+(b-1)*nBatchSize:b*nBatchSize), :);
    end; % end for
    
end % end function