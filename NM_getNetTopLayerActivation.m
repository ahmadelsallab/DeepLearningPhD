function [mTopLayerActivations] = NM_getNetTopLayerActivation(mCurrTrainMiniBatchData, NM_strNetParams, eMappingDirection, eMappingMode)
    % Get TopLayerActivations according to Net type
    switch(eMappingDirection)
        case 'BREADTH'                           
            switch(eMappingMode)
                case 'SVM_BREADTH_CLASSIFIER'
                    % Get Net activation (Feedforward) for top layer.
                    % NM_baseUnitActivation since no mapping is done yet
                    [mTopLayerActivations mAugActivationData] = NM_baseUnitActivation(mCurrTrainMiniBatchData, NM_strNetParams.cWeights); 
                    if(nPhase == nNumPhases)
                        mTopLayerActivations = [mTopLayerActivations mCurrTrainMiniBatchData];
                    end % end if-else
                otherwise
                    % Get Net activation (Feedforward) for top layer.
                    % NM_baseUnitActivation since no mapping is done yet
                    [mTopLayerActivations mAugActivationData] = NM_baseUnitActivation(mCurrTrainMiniBatchData, NM_strNetParams.cWeights);
            end % end switch-eMappingMode
        case 'DEPTH'
            switch(eMappingMode)
                case 'DEPTH_BASE_UNIT_MAPPING'
                    if(bMapping == 1)
                        [mTopLayerActivations tempActivationAugmented] = NM_compositeNetActivation(mCurrTrainMiniBatchData, NM_strNetParams.cUnitWeights);
                    else
                        % Get Net activation (Feedforward) for top layer.
                        % NM_baseUnitActivation since no mapping is done yet
                        [mTopLayerActivations mAugActivationData] = NM_baseUnitActivation(mCurrTrainMiniBatchData, NM_strNetParams.cWeights);                                        
                    end % end if
                case 'DEPTH_CASCADED_DATA_REPRESENTATION_MAPPING'
                    [mTopLayerActivations mAugActivationData] = NM_baseUnitActivation(mCurrTrainMiniBatchData, NM_strNetParams.cCascadedBaseUnitWeights);                                    
                otherwise
                    % Get Net activation (Feedforward) for top layer.
                    % NM_baseUnitActivation since no mapping is done yet
                    [mTopLayerActivations mAugActivationData] = NM_baseUnitActivation(mCurrTrainMiniBatchData, NM_strNetParams.cWeights);                                    
            end % end switch-eMappingMode
		case 'SAME'
			switch(eMappingMode)
				case 'ADAPTIVE'
				    % Get Net activation (Feedforward) for top layer.
                    % NM_baseUnitActivation since no mapping is done yet
                    [mTopLayerActivations mAugActivationData] = NM_baseUnitActivation(mCurrTrainMiniBatchData, NM_strNetParams.cWeights);
			end
    end % end switch-eMappingDirection

end % end function