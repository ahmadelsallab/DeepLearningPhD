% Function:
% Makes feed forward in the net and reports the error.
% Inputs:
% mBatchData: Input data. Matrix (nxm), where n is the number of examples and m is the features vector length
% mBatchTargets: Associated targets. Vector (nxl), where n is the number of examples and l is the number of target classes
% NM_strNetParams: Net parameters to feed forward
% bMapping: Is classifier mapping (re-use) enabled
% eMappingDirection: See CONFIG_setConfigParams
% eMappingMode: See CONFIG_setConfigParams
% nPhase: The current phase of classifier re-use (mapping)
% nNumPhases: Total number of phases for classifier re-use (mapping)
% eTestMode = 'EPOCH_ERR_CALC' (calculate error in each epoch) or 'ABSOLUTE_ERR_CALC' (normal err calculation) 
% nBitfieldLength: The bitfield length of the Raw features
% vChunkLength: Vector storing the boundaries of each features chunk
% vOffset: The bitfield boundaries corresponding to each chunk
% eFeaturesMode: Raw, Normal, Binary, Bitfield
% Output:
% nErr: The error rate
% vTargetOut: The output targets. This is a number not vector, representing
% the index at which maximum output occurs. Recall that target layer
% outputs are equivalent to probabilities that the output is the
% corresponding index.
function [nErr] = TST_computeReconstructionErr(mBatchData, NM_strNetParams, nBitfieldLength, vChunkLength, vOffset, eFeaturesMode)

    
    % Get mCurrBatchData size information
    [nNumExamples nNumFeatures nNumBatches]=size(mBatchData);

    
    % Loop on all batches in the mCurrBatchData
    nErrReconstruction = 0;
    for ctrBatch = 1 : nNumBatches
        
        % Get only the current batch data and targets
        mCurrBatchData = [mBatchData(:,:,ctrBatch)];
        
        % Convert to normal format of data
        if(strcmp(eFeaturesMode, 'Raw'))
            [mCurrBatchData] = DCONV_convertRawToBitfield(mCurrBatchData, nBitfieldLength, vChunkLength, vOffset);
        end

        
%           data = [batchdata(:,:,batch)];
%   data = [data ones(N,1)];
%   w1probs = 1./(1 + exp(-data*w1)); w1probs = [w1probs  ones(N,1)];
%   w2probs = 1./(1 + exp(-w1probs*w2)); w2probs = [w2probs ones(N,1)];
%   w3probs = 1./(1 + exp(-w2probs*w3)); w3probs = [w3probs  ones(N,1)];
%   w4probs = w3probs*w4; w4probs = [w4probs  ones(N,1)];
%   w5probs = 1./(1 + exp(-w4probs*w5)); w5probs = [w5probs  ones(N,1)];
%   w6probs = 1./(1 + exp(-w5probs*w6)); w6probs = [w6probs  ones(N,1)];
%   w7probs = 1./(1 + exp(-w6probs*w7)); w7probs = [w7probs  ones(N,1)];
%   dataout = 1./(1 + exp(-w7probs*w8));
%   err= err +  1/N*sum(sum( (data(:,1:end-1)-dataout).^2 )); 
        % Normal activation sequence, feeding non-augmented data
        % Feed Fwd                       
        [mTempNotAugActivationData mAugActivationData] = NM_neuralNetActivation(mCurrBatchData, NM_strNetParams.cWeights(1 : NM_strNetParams.nNumLayers - 1));
        %w4probs = w3probs*w4; w4probs = [w4probs  ones(N,1)];
        mAugActivationData{NM_strNetParams.nNumLayers} = mAugActivationData{NM_strNetParams.nNumLayers - 1} * NM_strNetParams.cWeights{NM_strNetParams.nNumLayers};
        %mAugActivationData{NM_strNetParams.nNumLayers} =
        %[mAugActivationData{NM_strNetParams.nNumLayers} ones(nNumExamples,
        %1)]; Augmentation shall be done inside NM_neuralNetActivation
        
        % Reconstruct
        [mTempNotAugActivationData mReconstructedBatchData] = NM_neuralNetActivation(mAugActivationData{NM_strNetParams.nNumLayers}, NM_strNetParams.cWeights(NM_strNetParams.nNumLayers + 1 : length(NM_strNetParams.cWeights)));

        % Compute the reconstruction error
        nErrReconstruction = nErrReconstruction + 1 /nNumExamples * (sum(sum( (mCurrBatchData - mReconstructedBatchData{NM_strNetParams.nNumLayers}(:, 1:end-1)).^2 )));
        
    end % end for-batches
     
    nErr = nErrReconstruction;

end % end function