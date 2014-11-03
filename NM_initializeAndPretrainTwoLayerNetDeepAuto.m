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
function [cWeights, mHiddenActivations] = NM_initializeAndPretrainTwoLayerNetDeepAuto(cWeights, vLayersWidths, nPrevNumLayers, ctrLayer,...
                                                                              bMapping, nPreTrainEpochs, mVisibleActivations,...                          
                                                                              hFidLog, eMappingMode, eMappingDirection, bDepthCascadedDataRepMode,...
                                                                              nBitfieldLength, vChunkLength, vOffset, eFeaturesMode, nNumLayers)
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

    mWeights     = 0.1*randn(nFirstLayerWidth, nSecodLayerWidth);
    vSecodLayerBiases  = zeros(1, nSecodLayerWidth);      

    % Adjust the 2-layer net with initialized weights
    [mHiddenActivations, mWeights, vSecodLayerBiases, vFirstLayerBiases] = NM_preTrainTwoLayerNetDeepAuto(mWeights, vSecodLayerBiases, mVisibleActivations, nPreTrainEpochs, nSecodLayerWidth, hFidLog, nBitfieldLength, vChunkLength, vOffset, eFeaturesMode, ctrLayer);

    % The following is equivalent to (nNumLayers = 4):
    % w1=[vishid; hidrecbiases];
    % w2=[hidpen; penrecbiases];
    % w3=[hidpen2; penrecbiases2];
    % w4=[hidtop; toprecbiases];
    % w5=[hidtop'; topgenbiases]; 
    % w6=[hidpen2'; hidgenbiases2]; 
    % w7=[hidpen'; hidgenbiases]; 
    % w8=[vishid'; visbiases];        
    cWeights{ctrLayer}=[mWeights; vSecodLayerBiases];
    cWeights{2*nNumLayers - (ctrLayer - 1)} = [mWeights'; vFirstLayerBiases];

