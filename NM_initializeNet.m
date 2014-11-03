% Function:
% Initializes all layers (units) of the newtwork to be trained.
% Initialization includes pre-training if needed.
% Inputs:
% ctrUnit: The index of the unit to be initialized
% bMapping: If mapping phase started or not
% eMappingMode: Configured mapping mode
% NM_strNetParams: The input NW params before update
% nPhase: The mapping phase
% hFidLog: The log file handle to log mapping phase status
% nFeatureVecLen: The length of the input features vector used in logging mapping phase data
% nPreTrainEpochs: The number of pre-training epochs if enabled
% nBitfieldLength: The bitfield length of the Raw features
% vChunkLength: Vector storing the boundaries of each features chunk
% vOffset: The bitfield boundaries corresponding to each chunk
% eFeaturesMode: Raw, Normal, Binary, Bitfield
% mTrainBatchData: Train data needed to pre-train network in case of pre-training is enabled
% Output:
% NM_strNetParams: The updated network parameters
function [NM_strNetParams] = NM_initializeNet(bMapping, eMappingMode, eMappingDirection, bDepthCascadedDataRepMode,...
                                              NM_strNetParams, nPhase, hFidLog, nFeatureVecLen, nPreTrainEpochs, mTrainBatchData,...
                                              nBitfieldLength, vChunkLength, vOffset, eFeaturesMode)

        % Check if DEPTH_BASE_UNIT_MAPPING and next phases are initialized
        % i.e. bMapping == 1
        if (bMapping == 1 && strcmp(eMappingMode, 'DEPTH_BASE_UNIT_MAPPING'))
            % No pretraining is required in this case
            NM_strNetParams.nNumLayers = nPhase + 1;
            for ctrUnit = 1 : NM_strNetParams.nNumLayers
                [NM_strNetParams] = NM_initializeAndPretrainDepthBaseUnit(NM_strNetParams, ctrUnit);
            end

        % Otherwise it's either other type of mapping or basic unit
        % training phase
        else
            % First visible data is the raw input
            mVisibleActivations = mTrainBatchData;

            for ctrLayer = 1 : (NM_strNetParams.nNumLayers)
                
                if	(ctrLayer==1)
                    fprintf(1,'Initializing Layer %d with RBM: %d-%d \n', ctrLayer, nFeatureVecLen, NM_strNetParams.vLayersWidths(ctrLayer));
                    fprintf(hFidLog,'Initializing Layer %d with RBM: %d-%d \n', ctrLayer, nFeatureVecLen,NM_strNetParams.vLayersWidths(ctrLayer));
                else
                    fprintf(1,'Initializing Layer %d with RBM: %d-%d \n', ctrLayer, NM_strNetParams.vLayersWidths(ctrLayer-1),NM_strNetParams.vLayersWidths(ctrLayer));
                    fprintf(hFidLog,'Initializing Layer %d with RBM: %d-%d \n', ctrLayer, NM_strNetParams.vLayersWidths(ctrLayer-1),NM_strNetParams.vLayersWidths(ctrLayer));
                end % end if-else
                
                % Pre-training is implicit in the NM_initializeAndPretrainNormalNet
                [NM_strNetParams.cWeights, mHiddenActivations] = NM_initializeAndPretrainTwoLayerNet(NM_strNetParams.cWeights, NM_strNetParams.vLayersWidths,...
                                                                                                     NM_strNetParams.nPrevNumLayers, ctrLayer, bMapping,...
                                                                                                     nPreTrainEpochs, mVisibleActivations, hFidLog,...
                                                                                                     eMappingMode, eMappingDirection, bDepthCascadedDataRepMode,...
                                                                                                     nBitfieldLength, vChunkLength, vOffset, eFeaturesMode);
                
                % Set visible data for RBM to use in next layer initialization
                mVisibleActivations = mHiddenActivations;		
            end
            
        end
end % end function