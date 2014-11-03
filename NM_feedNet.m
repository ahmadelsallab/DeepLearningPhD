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
% Output:
% nErr: The error rate
% vTargetOut: The output targets. This is a number not vector, representing
% the index at which maximum output occurs. Recall that target layer
% outputs are equivalent to probabilities that the output is the
% corresponding index.
% mTargetLayerProbs: The upper layer Soft-Max activations
function [nErr, vTargetOut, mTargetLayerProbs] = NM_feedNet(mBatchData, mBatchTargets, NM_strNetParams, bMapping, eMappingDirection, eMappingMode,...
                                               nPhase, nNumPhases, eTestMode)

    % Initialize correctly classified examples counter
    ctrCorrectlyClassified = 0;
    
    % Get mCurrBatchData size information
    [nNumExamples nNumFeatures nNumBatches]=size(mBatchData);
    
    % Initialize target out
    vTargetOut = [];
    
    % Loop on all batches in the mCurrBatchData
    for ctrBatch = 1 : nNumBatches
        
        % Get only the current batch data and targets
        mCurrBatchData = [mBatchData(:,:,ctrBatch)];
        mCurrBatchTarget = [mBatchTargets(:,:,ctrBatch)];

        switch (eTestMode)
            case 'EPOCH_ERR_CALC'
                % Calculate top layer activations according to the type of the net
                % Then calculate classification output (targetout)
                switch(eMappingDirection)
                    case 'BREADTH'

                        switch(eMappingMode) 
                            case 'SVM_BREADTH_CLASSIFIER'
                                if(nPhase == nNumPhases)
                                    % For SVM breadth mapping, we have to modify the upper layer to
                                    % include raw data before classification happens, since input
                                    % top SVM classification layer is just raw data
                                    mAugActivationData{NM_strNetParams.nNumLayers} = [mAugActivationData{NM_strNetParams.nNumLayers}(:, 1:end-1) mCurrBatchData(:,:,ctrBatch) ones(nNumExamples,1)];

                                    % Calculate classification output (targetout)
                                    mCurrBatchTargetOut = exp(mAugActivationData{NM_strNetParams.nNumLayers} * NM_strNetParams.mClassWeights);                            
                                end % end if
                            otherwise
                                % Normal activation sequence, feeding non-augmented data
                                % (augmentation happens insided NM_neuralNetActivation).
                                [mTempNotAugActivationData mAugActivationData] = NM_neuralNetActivation(mCurrBatchData, NM_strNetParams.cWeights);

                                % Calculate targetout
                                if(bMapping == 1)
                                    % 0.5 because the output is doubled due to classifier
                                    % replication
                                    mCurrBatchTargetOut = 0.5 * exp(mAugActivationData{NM_strNetParams.nNumLayers} * NM_strNetParams.mClassWeights);
                                else
                                    % Calculate classification output (targetout)
                                    mCurrBatchTargetOut = exp(mAugActivationData{NM_strNetParams.nNumLayers} * NM_strNetParams.mClassWeights);
                                end % end if
                        end % end switch eMappingMode

                    case 'DEPTH'
                        switch(eMappingMode)
                            case 'DEPTH_CASCADED_DATA_REPRESENTATION_MAPPING'
                                if(bMapping == 1)
                                    % Feed data first to the cascaded net
                                    [mTempNotAugActivationData mCurrBatchData] = NM_baseUnitActivation(mCurrBatchData, NM_strNetParams.cCascadedBaseUnitWeights);

                                    % Get the final activations after applying to
                                    % current net
                                    [mTempNotAugActivationData mAugActivationData] = NM_neuralNetActivation(mCurrBatchData, NM_strNetParams.cWeights);

                                    % Calculate classification output (targetout)
                                    mCurrBatchTargetOut = exp(mAugActivationData{NM_strNetParams.nNumLayers} * NM_strNetParams.mClassWeights);
                                else
                                    % Normal activation sequence, feeding non-augmented data
                                    % (augmentation happens insided NM_neuralNetActivation).                            
                                    [mTempNotAugActivationData mAugActivationData] = NM_neuralNetActivation(mCurrBatchData, NM_strNetParams.cWeights);

                                    % Calculate classification output (targetout)
                                    mCurrBatchTargetOut = exp(mAugActivationData{NM_strNetParams.nNumLayers} * NM_strNetParams.mClassWeights);
                                end % end if
                            case 'DEPTH_BASE_UNIT_MAPPING'
                                if(bMapping == 1)
                                    [mTempNotAugActivationData mAugActivationData] = NM_compositeNetActivation(mCurrBatchData, NM_strNetParams.cUnitWeights);
                                else
                                    % Normal activation sequence, feeding non-augmented data
                                    % (augmentation happens insided NM_neuralNetActivation).                            
                                    [mTempNotAugActivationData mAugActivationData] = NM_neuralNetActivation(mCurrBatchData, NM_strNetParams.cWeights);

                                    % Calculate classification output (targetout)
                                    mCurrBatchTargetOut = exp(mAugActivationData{NM_strNetParams.nNumLayers} * NM_strNetParams.mClassWeights);
                                end % end if
                            otherwise
                                % Normal activation sequence, feeding non-augmented data
                                % (augmentation happens insided NM_neuralNetActivation).                        
                                [mTempNotAugActivationData mAugActivationData] = NM_neuralNetActivation(mCurrBatchData, NM_strNetParams.cWeights);

                                % Calculate classification output (targetout)
                                mCurrBatchTargetOut = exp(mAugActivationData{NM_strNetParams.nNumLayers} * NM_strNetParams.mClassWeights);

                        end % end switch-eMappingMode
					case 'SAME'
                        switch(eMappingMode)
                            case 'ADAPTIVE'
							    % Normal activation sequence, feeding non-augmented data
                                % (augmentation happens insided NM_neuralNetActivation).                        
                                [mTempNotAugActivationData mAugActivationData] = NM_neuralNetActivation(mCurrBatchData, NM_strNetParams.cWeights);

                                % Calculate classification output (targetout)
                                mCurrBatchTargetOut = exp(mAugActivationData{NM_strNetParams.nNumLayers} * NM_strNetParams.mClassWeights);
						end
                end % end switch-eMappingDirection
                
            case 'ABSOLUTE_ERR_CALC'
                % Normal activation sequence, feeding non-augmented data
                % (augmentation happens insided NM_neuralNetActivation).                        
                [mTempNotAugActivationData mAugActivationData] = NM_neuralNetActivation(mCurrBatchData, NM_strNetParams.cWeights);

                % Calculate classification output (targetout)
                mCurrBatchTargetOut = exp(mAugActivationData{NM_strNetParams.nNumLayers} * NM_strNetParams.mClassWeights);                
        end % end switch-eTestMode

        % Normalize the output classification decision - Softmax
        nNumTargets = size(mBatchTargets, 2);        
        mCurrBatchTargetOut = mCurrBatchTargetOut./repmat(sum(mCurrBatchTargetOut,2), 1, nNumTargets);
        mTargetLayerProbs = mCurrBatchTargetOut;

        % Calculcate how many examples of this batch were correctly
        % classified
        % The max operation below is along the 2nd dimension of the batch
        % targets, so the output is a vector of length = nNumExamples
        [I J]=max(mCurrBatchTargetOut, [], 2); % J is the index where max. output is found in the Net targetout
        [I1 J1]=max(mCurrBatchTarget, [], 2); % J1 is the index where max. output is found in the desired target
        
        % Update total target out
        vTargetOut = [vTargetOut J];
        
        % The number of correctly classified examples is accumulated with
        % the number of examples in the current batch where the network
        % decision (targetout-->J) == the desired decision (target-->J1)
        ctrCorrectlyClassified = ctrCorrectlyClassified + length(find(J==J1)); 
        
    end % end for-batches
    
    % The error = Total number of examples - correctly classified ones
    nErr = (nNumExamples * nNumBatches - ctrCorrectlyClassified);
        

end % end function