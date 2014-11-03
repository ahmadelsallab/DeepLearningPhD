% Function:
% Initializes a 2-layer Net according to the mapping mode.
% Inputs:
% cWeights: The cell array of the Net weights, nNumLayersx{vLayerWidths(layer-1) X vLayerWidths(layer)}
% vLayersWidths: The vector of layers widths
% nPrevNumLayers: The number of layers of the network before mapping
% ctrLayer: The current counter of the layer to be initialized
% bMapping: If mapping phase started or not
% bEnablePretraining: If Pre-training is enabled or not
% nPreTrainEpochs: The number of pre-training epochs if enabled
% mVisibleActivations: The input data to the layer
% hFidLog: Log file handle
% eMappingMode: Mapping mode (see CONFIG_setConfigParams)
% eMappingDirection: Mapping direction (see CONFIG_setConfigParams)
% bDepthCascadedDataRepMode: Depth cascade representation mapping mode (see CONFIG_setConfigParams)
% nBitfieldLength: The bitfield length of the Raw features
% vChunkLength: Vector storing the boundaries of each features chunk
% vOffset: The bitfield boundaries corresponding to each chunk
% eFeaturesMode: Raw, Normal, Binary, Bitfield
% Output:
% cWeights: The updated weights cell arrays
% mHiddenActivation: The activations after pre-training if any. Used as
% inputs to next layer pre-training and initialization
function [cWeights, mHiddenActivations] = NM_initializeAndPretrainTwoLayerNet(cWeights, vLayersWidths, nPrevNumLayers, ctrLayer,...
                                                                              bMapping, nPreTrainEpochs, mVisibleActivations,...                          
                                                                              hFidLog, eMappingMode, eMappingDirection, bDepthCascadedDataRepMode,...
                                                                              nBitfieldLength, vChunkLength, vOffset, eFeaturesMode)
    % Initialize variables
    mHiddenActivations = [];
    
    % Get the sizes of the layer input activation data    
    if(strcmp(eFeaturesMode, 'Raw') & ctrLayer == 1)
        nFirstLayerWidth = nBitfieldLength;
    else
        nFirstLayerWidth = size(mVisibleActivations, 2);
    end
    
    % Get the size of the 
    nSecodLayerWidth = vLayersWidths(ctrLayer);

    if bMapping == 1
        mOrigLayerWeightsWithoutBias = cWeights{ctrLayer}(1:end-1,:);
        switch(eMappingDirection)
            
            case 'BREADTH'
                
                switch(eMappingMode)
                    
                    case 'NN_BREADTH_CLASSIFIER_MAPPING'
                        
                        if (ctrLayer == 1)
                            mWeights        = [mOrigLayerWeightsWithoutBias                                                 mOrigLayerWeightsWithoutBias];
                        else
                            
                            mWeights        = [mOrigLayerWeightsWithoutBias                                                 zeros(size(mOrigLayerWeightsWithoutBias,1), size(mOrigLayerWeightsWithoutBias,2)); 
                                               zeros(size(mOrigLayerWeightsWithoutBias,1),...
                                                     size(mOrigLayerWeightsWithoutBias,2))                                  mOrigLayerWeightsWithoutBias];
                        end	
                        vSecodLayerBiases	= [cWeights{ctrLayer}(end,:)                                                cWeights{ctrLayer}(end,:)];
                        
                    case 'SEMI_RANDOM_BREADTH_CLASSIFIER_MAPPING'
                        
                        mWeights            = [mOrigLayerWeightsWithoutBias                                                 0.1*randn(size(mOrigLayerWeightsWithoutBias,1),...
                                                                                                                                  size(mOrigLayerWeightsWithoutBias,2))];
                        vSecodLayerBiases   = [cWeights{ctrLayer}(end,:)                                                cWeights{ctrLayer}(end,:)];
                        
                    case 'FULL_RANDOM_BREADTH_CLASSIFIER_MAPPING'
                        
                        mWeights            = [0.1*randn(size(mOrigLayerWeightsWithoutBias,1),size(mOrigLayerWeightsWithoutBias,2)) 0.1*randn(size(mOrigLayerWeightsWithoutBias,1), size(mOrigLayerWeightsWithoutBias,2))];
                        vSecodLayerBiases   = [0.1*randn(1,size(mOrigLayerWeightsWithoutBias,2))                                0.1*randn(1,size(mOrigLayerWeightsWithoutBias,2))];
                        
                    case 'SVM_BREADTH_CLASSIFIER'
                        
                        if (ctrLayer == 1)
                            mWeights        = [mOrigLayerWeightsWithoutBias                                                 mOrigLayerWeightsWithoutBias];
                        else
                            
                            mWeights        = [mOrigLayerWeightsWithoutBias                                                 zeros(size(mOrigLayerWeightsWithoutBias,1), size(mOrigLayerWeightsWithoutBias,2)); 
                                               zeros(size(mOrigLayerWeightsWithoutBias,1),...
                                                     size(mOrigLayerWeightsWithoutBias,2))                                  mOrigLayerWeightsWithoutBias];
                        end	
                        vSecodLayerBiases	= [cWeights{ctrLayer}(end,:)                                                    cWeights{ctrLayer}(end,:)];
                        
                    case 'WEAK_BREADTH_CLASSIFIER_MAPPING'
                        
                        if (ctrLayer == 1)
                            mWeights        = [mOrigLayerWeightsWithoutBias                                                 zeros(size(mOrigLayerWeightsWithoutBias,1), size(mOrigLayerWeightsWithoutBias,2))];
                        else
                            mWeights        = [mOrigLayerWeightsWithoutBias                                                 zeros(size(mOrigLayerWeightsWithoutBias,1), size(mOrigLayerWeightsWithoutBias,2)); zeros(size(mOrigLayerWeightsWithoutBias,1), size(mOrigLayerWeightsWithoutBias,2)) zeros(size(mOrigLayerWeightsWithoutBias,1), size(mOrigLayerWeightsWithoutBias,2))];
                        end
                        vSecodLayerBiases   = [cWeights{ctrLayer}(end,:)                                                    zeros(1,size(mOrigLayerWeightsWithoutBias,2))];                        
                end % end switch 'BREADTH'
                % end 'BREADTH' case
                
            case 'DEPTH'
                
                switch(eMappingMode)
                    
                    case 'NN_DEPTH_CLASSIFIER_MAPPING'
                        if  (ctrLayer == nPrevNumLayers + 1)
                            mWeights            = 0.1*randn(nFirstLayerWidth, nSecodLayerWidth);
                            vSecodLayerBiases   = zeros(1, nSecodLayerWidth);

                        else
                            mWeights            = cWeights{mod(ctrLayer-1, nPrevNumLayers) + 1}(1:end-1,:);
                            vSecodLayerBiases   = cWeights{mod(ctrLayer-1, nPrevNumLayers) + 1}(end,:);

                        end      
                        
                    case 'DEPTH_CASCADED_DATA_REPRESENTATION_MAPPING'
                        % Normal random initialization just as if no mapping
                        if(ctrLayer == 1)
                            nFirstLayerWidth    = size(cWeights{nPrevNumLayers}, 2);
                            mWeights            = 0.1*randn(nFirstLayerWidth, nSecodLayerWidth);                    
                            vSecodLayerBiases   = zeros(1, nSecodLayerWidth);
                        else
                            switch(bDepthCascadedDataRepMode)
                                case 'REPLICATED' 
                                    mWeights = mOrigLayerWeightsWithoutBias;
                                    vSecodLayerBiases = cWeights{ctrLayer}(end,:);                                    
                                case 'RANDOMIZE'
                                    mWeights     = 0.1*randn(nFirstLayerWidth, nSecodLayerWidth);                    
                                    vSecodLayerBiases  = zeros(1, nSecodLayerWidth);
                            end % end switch 
                        end    
                        
                end % end switch 'DEPTH'
            % end 'DEPTH' case  
            case 'SAME'
				switch(eMappingMode)
					case 'ADAPTIVE'
						% Keep same weights
						cWeights = cWeights;
						mWeights = mOrigLayerWeightsWithoutBias;
						vSecodLayerBiases = cWeights{ctrLayer}(end,:);
						
				end % end switch 'ADAPTIVE'
        end % end switch

    else % no mapping yet, so normal base unit training        
        mWeights     = 0.1*randn(nFirstLayerWidth, nSecodLayerWidth);
        vSecodLayerBiases  = zeros(1, nSecodLayerWidth);  
    end % end if(bMapping)
    

    % Adjust the 2-layer net with initialized weights
    [mHiddenActivations, mWeights, vSecodLayerBiases] = NM_preTrainTwoLayerNet(mWeights, vSecodLayerBiases, mVisibleActivations, nPreTrainEpochs, nSecodLayerWidth, hFidLog, nBitfieldLength, vChunkLength, vOffset, eFeaturesMode);

        
    cWeights{ctrLayer}=[mWeights; vSecodLayerBiases];

