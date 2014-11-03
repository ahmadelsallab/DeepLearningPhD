% Function:
% Makes feed forward in the net and reports the upper layer probs.Recall that target layer
% outputs are equivalent to probabilities that the output is the corresponding index.
% Inputs:
% mBatchData: Input data. Matrix (nxm), where n is the number of examples and m is the features vector length
% NM_strNetParams: Net parameters to feed forward
% nBitfieldLength: The bitfield length of the Raw features
% vChunkLength: Vector storing the boundaries of each features chunk
% vOffset: The bitfield boundaries corresponding to each chunk
% eFeaturesMode: Raw, Normal, Binary, Bitfield
% Output:
% vTargetOut: The output classification target (nx1)where n is the number of examples
function [vTargetOut] = TST_computeUpperLayerTargets(mBatchData, NM_strNetParams, nBitfieldLength, vChunkLength, vOffset, eFeaturesMode)

    
    % Get mCurrBatchData size information
    [nNumExamples nNumFeatures nNumBatches]=size(mBatchData);
    
    vTargetOut = [];
          
    % Loop on all batches in the mCurrBatchData
    for ctrBatch = 1 : nNumBatches
        
        % Get only the current batch data and targets
        mCurrBatchData = [mBatchData(:,:,ctrBatch)];
        
        if(strcmp(eFeaturesMode, 'Raw'))
            [mCurrBatchData] = DCONV_convertRawToBitfield(mCurrBatchData, nBitfieldLength, vChunkLength, vOffset);
        end

        % Normal activation sequence, feeding non-augmented data
        % (augmentation happens insided NM_neuralNetActivation).                        
        [mTempNotAugActivationData mAugActivationData] = NM_neuralNetActivation(mCurrBatchData, NM_strNetParams.cWeights);

        % Calculate classification output (targetout)
        mCurrBatchTargetOut = exp(mAugActivationData{NM_strNetParams.nNumLayers} * NM_strNetParams.mClassWeights); 

        % Normalize the output classification decision - Softmax
        
        
%        mBitfieldOutput(:,:,ctrBatch) = mCurrBatchTargetOut./repmat(sum(mCurrBatchTargetOut,2), 1, nNumTargets);
        
        [I J]=max(mCurrBatchTargetOut, [], 2); % J is the index where max. output is found in the Net targetout
        
        % Update total target out
        vTargetOut = [vTargetOut; J];
      
    end % end for-batches
    
end % end function