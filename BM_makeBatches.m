% Function:
% Converts the test and train sets into batches
% Inputs:
% mTestFeatures: Test features. Matrix (nxm), where n is the number of examples and m is the features vector length
% mTestTargets: Test targetss. Matrix (nxl), where n is the number of examples and l is the number of target classes
% mTrainFeatures: Train features. Matrix (nxm), where n is the number of examples and m is the features vector length
% mTrainTargets: Train targets. Matrix (nxl), where n is the number of examples and l is the number of target classes
% nMaxFeaturesRange: The maximum dynamic range of the numeric value of each feature
% bAutoLabel: Flag to indicate if automatic (unsupervised) labeling is used
% mAutoLabels: The automatic labels
% nBatchSize: The batch size in number of examples
% Output:
% mTrainBatchTargets, mTrainBatchData, mTestBatchTargets, mTestBatchData: Matrices (nBatchSize X nNumFeatures X nNumBatches) similar to input ones
function [mTrainBatchTargets, mTrainBatchData, mTestBatchTargets, mTestBatchData] = BM_makeBatches(mTestFeatures, mTestTargets, mTrainFeatures, mTrainTargets, nMaxFeaturesRange, bAutoLabel, mAutoLabels, nBatchSize)

    nNumTargets = size(mTrainTargets, 2);
    % Mark targets according to auto labels
    if(bAutoLabel == 1)
        for i = 1 : size(mAutoLabels, 2)
            for j = 1 : nNumTargets
                if(mAutoLabels(i)==j)
                    mTrainTargets(i,j) = 1;
                else
                    mTrainTargets(i,j) = 0;
                end
            end
        end
    end

    % Training data
    mTrainFeatures = mTrainFeatures/nMaxFeaturesRange;
    [mTrainBatchData, mTrainBatchTargets] = BM_randomizeBatchData(nBatchSize, mTrainFeatures, mTrainTargets);
    

    % Test data
    mTestFeatures = mTestFeatures/nMaxFeaturesRange;

    mTestTargets = mTestTargets/nMaxFeaturesRange;
    [mTestBatchData, mTestBatchTargets] = BM_randomizeBatchData(nBatchSize, mTestFeatures, mTestTargets);


    %%% Reset random seeds 
    rand('state',sum(100*clock)); 
    randn('state',sum(100*clock)); 

end % end function