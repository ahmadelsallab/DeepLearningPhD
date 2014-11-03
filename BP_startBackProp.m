% Function:
% Performs back propagation for the curent epoch on the all batches
% Inputs:
% CONFIG_strParams: Configuration parameters
% NM_strNetParams: Net parameters to update
% mTrainBatchData, mTrainBatchTargets: Training set. See BM_makeBatches
% nEpoch: The current iteration number
% vLayersSize: The sizes of net layers
% cPrevWeights: Weights of the previous learning phase (not previous epoch)
% mPrevClassWeights: Class weights of the previous learning phase (not previous epoch)
% nPhase, nNumPhases, bMapping: See CLS_fineTuneAndClassifyDNN
% sFeaturesFileName: String of the input txt file
% nBitfieldLength: The bitfield length of the Raw features
% vChunkLength: Vector storing the boundaries of each features chunk
% vOffset: The bitfield boundaries corresponding to each chunk
% eFeaturesMode: Raw, Normal, Binary, Bitfield
% Output:
% cWeights: Updated weights of each layer of unit
% NM_strNetParams: Updated Net parameters
function [cWeights, NM_strNetParams] = BP_startBackProp(CONFIG_strParams, NM_strNetParams, mTrainBatchData, mTrainBatchTargets,...
                                                        nEpoch, vLayersSize, cPrevWeights, mPrevClassWeights, nPhase, nNumPhases, bMapping,...
                                                        nBitfieldLength, vChunkLength, vOffset, eFeaturesMode)
        
        % Obtain training set sizes
        [nNumExamplesPerBatch nNumFeaturesPerExample nNumBatches] = size(mTrainBatchData);
        
        if(strcmp(eFeaturesMode, 'Raw'))
            nNumFeaturesPerExample = nBitfieldLength;
        end
        
        % Initialize variables
        nMiniBatchIndex = 0;
        
        cWeights = NM_strNetParams.cWeights;
        
        % Start the loop on the number of mini batches. The loop terminates
        % when all minibatches inside numbatches end. So loop end =
        % nNumBatches/nBPNumExamplesInMiniBatch
        for nBatchNum = 1 : nNumBatches / CONFIG_strParams.nBPNumExamplesInMiniBatch

            fprintf(1,'epoch %d batch %d\r',nEpoch,nBatchNum);

            %%%%%%%%%%% COMBINE CONFIG_strParams.nBPNumExamplesInMiniBatch MINIBATCHES INTO 1 LARGER MINIBATCH %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Get the next minibatch index
            nMiniBatchIndex = nMiniBatchIndex + 1; 

            % Combine minibatches into one data/target matrices
            mCurrTrainMiniBatchData=[];
            mCurrTrainMiniBatchTargets=[]; 
            for ctrMiniBatch = 1 : CONFIG_strParams.nBPNumExamplesInMiniBatch

                % Augment Batch data
                mCurrTrainMiniBatchData = [mCurrTrainMiniBatchData 
                                           mTrainBatchData(:,:,(nMiniBatchIndex-1) * CONFIG_strParams.nBPNumExamplesInMiniBatch + ctrMiniBatch)]; 
                
                % Augment Batch targets
                mCurrTrainMiniBatchTargets = [mCurrTrainMiniBatchTargets
                                              mTrainBatchTargets(:,:,(nMiniBatchIndex-1)*CONFIG_strParams.nBPNumExamplesInMiniBatch + ctrMiniBatch)];
            end 
            
            % Convert to bitfield if needed
            if(strcmp(eFeaturesMode, 'Raw'))
                [mCurrTrainMiniBatchData] = DCONV_convertRawToBitfield(mCurrTrainMiniBatchData, nBitfieldLength, vChunkLength, vOffset);
            end

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% START MIMIZER %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%%%%%%%%%%%% PERFORM CONJUGATE GRADIENT WITH 3 LINESEARCHES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%


            if (bMapping == 0)
                % First update top-level weights holding other weights fixed. 
                if nEpoch < CONFIG_strParams.nBPNumEpochsForUpperLayerTraining  
                    
                    % Get Net activation (Feedforward) for top layer.
                    % NM_baseUnitActivation since no mapping is done yet
                    [mTopLayerActivations mAugActivationData] = NM_baseUnitActivation(mCurrTrainMiniBatchData, NM_strNetParams.cWeights);

                    switch (CONFIG_strParams.eMappingDirection)
                        case 'DEPTH'
                            % Matrix to vector operation for both weights and
                            % layers sizes to be minimized over
                            vWeightsToMinimize = [NM_strNetParams.mClassWeights(:)']';
                            vLayersSizeToMinimize = [vLayersSize(NM_strNetParams.nNumLayers + 1); vLayersSize(NM_strNetParams.nNumLayers + 2)];
                            
                            % Start the minimizer                            
                            [X, fX] = minimize(vWeightsToMinimize, 'CG_CLASSIFY_INIT', CONFIG_strParams.nMaxIterCGMinimizer, vLayersSizeToMinimize, mTopLayerActivations,mCurrTrainMiniBatchTargets);

                            % Update class weights
                            % Vector to matrix operation to reverse the
                            % operation before minimizer call
                            NM_strNetParams.mClassWeights = reshape(X,vLayersSize(NM_strNetParams.nNumLayers + 1) + 1,vLayersSize(NM_strNetParams.nNumLayers + 2));                                                            
                            
                        case 'BREADTH'
                            switch(CONFIG_strParams.eMappingMode)
                                case 'SVM_TOP_LEVEL_INTEGRATED_CLASSIFIER_MAPPING'
                                    
                                    % Train SVM's on top of the current
                                    % network. The training data for the
                                    % SVM is the top layer activations of
                                    % the current net.
                                    [NM_strNetParams.mClassWeights] = CLS_trainSVMOnTopDNN(mTopLayerActivations, mCurrTrainMiniBatchTargets);
                                    
                                otherwise
                                    % Matrix to vector operation for both weights and
                                    % layers sizes to be minimized over
                                    vWeightsToMinimize = [NM_strNetParams.mClassWeights(:)']';
                                    vLayersSizeToMinimize = [vLayersSize(NM_strNetParams.nNumLayers + 1); vLayersSize(NM_strNetParams.nNumLayers + 2)];
                                    
                                    % Start the minimizer
                                    [X, fX] = minimize(vWeightsToMinimize, 'CG_CLASSIFY_INIT', CONFIG_strParams.nMaxIterCGMinimizer, vLayersSizeToMinimize, mTopLayerActivations,mCurrTrainMiniBatchTargets);

                                    % Update class weights
                                    % Vector to matrix operation to reverse the
                                    % operation before minimizer call
                                    NM_strNetParams.mClassWeights = reshape(X,vLayersSize(NM_strNetParams.nNumLayers + 1) + 1,vLayersSize(NM_strNetParams.nNumLayers + 2));                                
							
                            end % end switch eMappingMode
						case 'SAME'
							switch(CONFIG_strParams.eMappingMode)
								case 'ADAPTIVE'
								    % Matrix to vector operation for both weights and
                                    % layers sizes to be minimized over
                                    vWeightsToMinimize = [NM_strNetParams.mClassWeights(:)']';
                                    vLayersSizeToMinimize = [vLayersSize(NM_strNetParams.nNumLayers + 1); vLayersSize(NM_strNetParams.nNumLayers + 2)];
                                    
                                    % Start the minimizer
                                    [X, fX] = minimize(vWeightsToMinimize, 'CG_CLASSIFY_INIT', CONFIG_strParams.nMaxIterCGMinimizer, vLayersSizeToMinimize, mTopLayerActivations,mCurrTrainMiniBatchTargets);

                                    % Update class weights
                                    % Vector to matrix operation to reverse the
                                    % operation before minimizer call
                                    NM_strNetParams.mClassWeights = reshape(X,vLayersSize(NM_strNetParams.nNumLayers + 1) + 1,vLayersSize(NM_strNetParams.nNumLayers + 2));	
							end
                    end % end switch eMappingDirection
                else % nEpoch < nNumTrainUpperLayerEpochs
                    switch(CONFIG_strParams.eMappingDirection)
                        case 'DEPTH'
                            
                            % Matrix to vector operation for both weights and
                            % layers sizes to be minimized over
                            vWeightsToMinimize = [];
                            for(layer = 1 : NM_strNetParams.nNumLayers)
                                vWeightsToMinimize = [vWeightsToMinimize NM_strNetParams.cWeights{layer}(:)'];
                            end
                            vWeightsToMinimize = [vWeightsToMinimize NM_strNetParams.mClassWeights(:)'];
                            vWeightsToMinimize = vWeightsToMinimize';

                            vLayersSizeToMinimize = [];
                            for(layer = 1 : (NM_strNetParams.nNumLayers+2))
                                vLayersSizeToMinimize = [vLayersSizeToMinimize; vLayersSize(layer)];
                            end

                            % Start the minimizer
                            [X, fX] = minimize(vWeightsToMinimize,'CG_CLASSIFY',CONFIG_strParams.nMaxIterCGMinimizer,vLayersSizeToMinimize, mCurrTrainMiniBatchData,...
                                               mCurrTrainMiniBatchTargets, 0, 0, 0, 0);

                            % Update class weights
                            % Vector to matrix operation to reverse the
                            % operation before minimizer call

                            nOffset = 0;
                            
                            % Net weights
                            for(layer = 1 : NM_strNetParams.nNumLayers)
                                NM_strNetParams.cWeights{layer} = reshape(X(nOffset+1:nOffset+(vLayersSize(layer)+1)*vLayersSize(layer+1)), vLayersSize(layer)+1, vLayersSize(layer+1));
                                nOffset = nOffset + (vLayersSize(layer)+1)*vLayersSize(layer+1);
                            end          
                            
                            % Class weights
                            NM_strNetParams.mClassWeights =...
                                reshape(X(nOffset+1:nOffset+(vLayersSize(NM_strNetParams.nNumLayers+1)+1) * vLayersSize(NM_strNetParams.nNumLayers+2)),...
                                                           vLayersSize(NM_strNetParams.nNumLayers+1)+1, vLayersSize(NM_strNetParams.nNumLayers+2));
                            
                        case 'BREADTH'
                            switch(CONFIG_strParams.eMappingMode)
                                case 'SVM_TOP_LEVEL_INTEGRATED_CLASSIFIER_MAPPING'
                                    
                                    % Get Net activation (Feedforward) for top layer.
                                    % NM_baseUnitActivation since no mapping is done yet
                                    [mTopLayerActivations mAugActivationData] = NM_baseUnitActivation(mCurrBatchData, NM_strNetParams.cWeights);
                                    
                                    % Train SVM's on top of the current
                                    % network. The training data for the
                                    % SVM is the top layer activations of
                                    % the current net.
                                    [NM_strNetParams.mClassWeights] = CLS_trainSVMOnTopDNN(mTopLayerActivations, mCurrTrainMiniBatchTargets);
                                    
                                otherwise
                                    % Matrix to vector operation for both weights and
                                    % layers sizes to be minimized over
                                    vWeightsToMinimize = [];
                                    for(layer = 1 : NM_strNetParams.nNumLayers)
                                        vWeightsToMinimize = [vWeightsToMinimize NM_strNetParams.cWeights{layer}(:)'];
                                    end
                                    vWeightsToMinimize = [vWeightsToMinimize NM_strNetParams.mClassWeights(:)'];
                                    vWeightsToMinimize = vWeightsToMinimize';

                                    vLayersSizeToMinimize = [];
                                    for(layer = 1 : (NM_strNetParams.nNumLayers+2))
                                        vLayersSizeToMinimize = [vLayersSizeToMinimize; vLayersSize(layer)];
                                    end

                                    % Start the minimizer                                                                        
                                    [X, fX] = minimize(vWeightsToMinimize,'CG_CLASSIFY',CONFIG_strParams.nMaxIterCGMinimizer, vLayersSizeToMinimize,...
                                                       mCurrTrainMiniBatchData, mCurrTrainMiniBatchTargets, 0, 0, 0, 0);

                                    % Update class weights
                                    % Vector to matrix operation to reverse the
                                    % operation before minimizer call
                                    
                                    nOffset = 0;
                                    
                                    % Net weights
                                    for(layer = 1 : NM_strNetParams.nNumLayers)
                                        NM_strNetParams.cWeights{layer} = reshape(X(nOffset+1:nOffset+(vLayersSize(layer)+1) * vLayersSize(layer+1)), vLayersSize(layer)+1, vLayersSize(layer+1));
                                        nOffset = nOffset + (vLayersSize(layer)+1)*vLayersSize(layer+1);
                                    end                               

                                    % Class weights
                                    NM_strNetParams.mClassWeights =...
                                        reshape(X(nOffset+1:nOffset+(vLayersSize(NM_strNetParams.nNumLayers+1)+1) * vLayersSize(NM_strNetParams.nNumLayers+2)),...
                                                vLayersSize(NM_strNetParams.nNumLayers+1)+1, vLayersSize(NM_strNetParams.nNumLayers+2));

                            end % end switch eMappingMode
						case 'SAME'
							switch(CONFIG_strParams.eMappingMode)
								case 'ADAPTIVE'
								    % Matrix to vector operation for both weights and
                                    % layers sizes to be minimized over
                                    vWeightsToMinimize = [];
                                    for(layer = 1 : NM_strNetParams.nNumLayers)
                                        vWeightsToMinimize = [vWeightsToMinimize NM_strNetParams.cWeights{layer}(:)'];
                                    end
                                    vWeightsToMinimize = [vWeightsToMinimize NM_strNetParams.mClassWeights(:)'];
                                    vWeightsToMinimize = vWeightsToMinimize';

                                    vLayersSizeToMinimize = [];
                                    for(layer = 1 : (NM_strNetParams.nNumLayers+2))
                                        vLayersSizeToMinimize = [vLayersSizeToMinimize; vLayersSize(layer)];
                                    end

                                    % Start the minimizer                                                                        
                                    [X, fX] = minimize(vWeightsToMinimize,'CG_CLASSIFY',CONFIG_strParams.nMaxIterCGMinimizer, vLayersSizeToMinimize,...
                                                       mCurrTrainMiniBatchData, mCurrTrainMiniBatchTargets, 0, 0, 0, 0);

                                    % Update class weights
                                    % Vector to matrix operation to reverse the
                                    % operation before minimizer call
                                    
                                    nOffset = 0;
                                    
                                    % Net weights
                                    for(layer = 1 : NM_strNetParams.nNumLayers)
                                        NM_strNetParams.cWeights{layer} = reshape(X(nOffset+1:nOffset+(vLayersSize(layer)+1) * vLayersSize(layer+1)), vLayersSize(layer)+1, vLayersSize(layer+1));
                                        nOffset = nOffset + (vLayersSize(layer)+1)*vLayersSize(layer+1);
                                    end                               

                                    % Class weights
                                    NM_strNetParams.mClassWeights =...
                                        reshape(X(nOffset+1:nOffset+(vLayersSize(NM_strNetParams.nNumLayers+1)+1) * vLayersSize(NM_strNetParams.nNumLayers+2)),...
                                                vLayersSize(NM_strNetParams.nNumLayers+1)+1, vLayersSize(NM_strNetParams.nNumLayers+2));
							end
                    end % end switch eMappingDirection                                        
                end
            else
                if nEpoch <= CONFIG_strParams.nBPNumTrainUpperLayerEpochs   % First update top-level weights holding other weights fixed. 
                    
                    % Get TopLayerActivations according to Net type
                    [mTopLayerActivations] = NM_getNetTopLayerActivation(mCurrTrainMiniBatchData, NM_strNetParams,...
                                                                         CONFIG_strParams.eMappingDirection, CONFIG_strParams.eMappingMode); 
                    
                    % Start Minimization
                    if CONFIG_strParams.bBPEnablePenaltyIter == 1
                        % Penalty iterations enabled
                        for i = 1 : CONFIG_strParams.nNumPenaltyIterations	
                            
                            % Matrix to vector conversion of weights to be
                            % minimized
                            vWeightsToMinimize = [NM_strNetParams.mClassWeights(:)']';
                            vLayersSizeToMinimize = [vLayersSize(NM_strNetParams.nNumLayers + 1); vLayersSize(NM_strNetParams.nNumLayers + 2)];

                            % Get PREVIOUS net activations
                            [mPrevTopLayerActivations mAugActivationData] = NM_baseUnitActivation(mCurrTrainMiniBatchData, cPrevWeights);                            
                            
                            % Perform penalty iterations                            
                            switch(CONFIG_strParams.eBarrierType)
                                case 'LOG_BARRIER'
                                    beta = CONFIG_strParams.nLogBarrierMinimizerBeta;
                                    lambda = CONFIG_strParams.nDynMinimizerLambda;
                                    
                                    % Start minimizer
                                    [X, fX] = minimize(vWeightsToMinimize,'CG_CLASSIFY_INIT_CONSTRAINED_LOG',CONFIG_strParams.nMaxIterCGMinimizer, vLayersSizeToMinimize,...
                                                       mTopLayerActivations,mCurrTrainMiniBatchTargets, mPrevTopLayerActivations, mPrevClassWeights, beta);
                                    
                                    % Do vector to matrix conversion
                                    NM_strNetParams.mClassWeights = reshape(X,vLayersSize(NM_strNetParams.nNumLayers+1)+1,vLayersSize(NM_strNetParams.nNumLayers+2));

                                    % dynamic beta
                                    if (CONFIG_strParams.bDynamicPenaltyBarrier == 1)
                                        beta = lambda*beta;
                                    end                                    
                                case 'SQUARE_BARRIER'
                                    alpha  = CONFIG_strParams.nSquareBarrierMinimizerAlpha;
                                    lambda = CONFIG_strParams.nDynMinimizerLambda;
                                    
                                    % Start minimizer
                                    [X, fX] = minimize(vWeightsToMinimize,'CG_CLASSIFY_INIT_CONSTRAINED_SQUARE',CONFIG_strParams.nMaxIterCGMinimizer,vLayersSizeToMinimize,...
                                                       mTopLayerActivations,mCurrTrainMiniBatchTargets, mPrevTopLayerActivations, mPrevClassWeights, alpha);
                                                   
                                    % Do vector to matrix conversion    
                                    NM_strNetParams.mClassWeights = reshape(X,vLayersSize(NM_strNetParams.nNumLayers+1)+1,vLayersSize(NM_strNetParams.nNumLayers+2));
                                    
                                    if (CONFIG_strParams.bDynamicPenaltyBarrier == 1)
                                        alpha = alpha*lambda;
                                    end                      
                                    
                            end % end switch
                        end
                    else
                        % Penalty iterations disabled
                        
                        % Matrix to vector conversion of weights to be
                        % minimized
                        vWeightsToMinimize = [NM_strNetParams.mClassWeights(:)']';
                        vLayersSizeToMinimize = [vLayersSize(NM_strNetParams.nNumLayers+1); vLayersSize(NM_strNetParams.nNumLayers+2)];
                        
                        % Start minimizer
                        [X, fX] = minimize(vWeightsToMinimize,'CG_CLASSIFY_INIT',CONFIG_strParams.nMaxIterCGMinimizer,vLayersSizeToMinimize, mTopLayerActivations, mCurrTrainMiniBatchTargets);
                        
                        % Do vector to matrix conversion
                        NM_strNetParams.mClassWeights = reshape(X,vLayersSize(NM_strNetParams.nNumLayers+1)+1,vLayersSize(NM_strNetParams.nNumLayers+2));
                    end
                else
                    if nEpoch < CONFIG_strParams.nNumTrain_N_UpperLayerEpochs
                        % Get TopLayerActivations according to Net type
                        [mTopLayerActivations] = NM_getNetTopLayerActivation(mCurrTrainMiniBatchData, NM_strNetParams, CONFIG_strParams.eMappingDirection, CONFIG_strParams.eMappingMode);
                        
                        if CONFIG_strParams.bBPEnablePenaltyIter == 1
                            for i = 1 : CONFIG_strParams.nNumPenaltyIterations
                                
                                % Do matrix to vecotr conversion of weigths
                                % to be minimized
                                vWeightsToMinimize = [];
                                vLayersSizeToMinimize = [];

                                for layer = ((NM_strNetParams.nNumLayers - CONFIG_numTrainUpperLayer)+1) : NM_strNetParams.nNumLayers
                                    vWeightsToMinimize = [vWeightsToMinimize NM_strNetParams.cWeights{layer}(:)'];
                                    vLayersSizeToMinimize = [vLayersSizeToMinimize; vLayersSize(layer)];
                                end
                                vWeightsToMinimize = [vWeightsToMinimize NM_strNetParams.mClassWeights(:)']';
                                vLayersSizeToMinimize = [vLayersSizeToMinimize; vLayersSize(NM_strNetParams.nNumLayers+1); vLayersSize(NM_strNetParams.nNumLayers+2)];
                                
                                % Get PREVIOUS net activations
                                [mPrevTopLayerActivations mAugActivationData] = NM_baseUnitActivation(mCurrTrainMiniBatchData, cPrevWeights); 
                                
                                alpha = CONFIG_minimizerAlpha;
                                lambda = CONFIG_minimizerLambda;

                                if (~strcmp(CONFIG_strParams.MappingMode, 'DEPTH_BASE_UNIT_MAPPING'))
                                    [X, fX] = minimize(vWeightsToMinimize,'CG_CLASSIFY_N_Layers_CONSTRAINED',CONFIG_strParams.nMaxIterCGMinimizer,vLayersSizeToMinimize,...
                                                       mTopLayerActivations{NM_strNetParams.nNumLayers-CONFIG_strParams.nNumTrain_N_UpperLayerEpochs},mCurrTrainMiniBatchTargets,...
                                                       mPrevTopLayerActivations{end-CONFIG_strParams.nNumTrain_N_UpperLayerEpochs}, cPrevWeights, mPrevClassWeights, alpha,...
                                                       CONFIG_strParams.MappingMode, 0, 0, 0);
                                else
                                    [X, fX] = minimize(vWeightsToMinimize,'CG_CLASSIFY_N_Layers_CONSTRAINED',CONFIG_strParams.nMaxIterCGMinimizer, vLayersSizeToMinimize,...
                                                       mTopLayerActivations{NM_strNetParams.nNumLayers-CONFIG_strParams.nNumTrain_N_UpperLayerEpochs}, mCurrTrainMiniBatchTargets,...
                                                       mPrevTopLayerActivations{end-CONFIG_strParams.nNumTrain_N_UpperLayerEpochs}, NM_baseUnitWeights, mPrevClassWeights, alpha,...
                                                       CONFIG_strParams.MappingMode, NM_strNetParams.cUnitWeights{NM_strNetParams.nNumLayers-CONFIG_strParams.nNumTrain_N_UpperLayerEpochs},...
                                                       NM_strNetParams.cWeights{NM_strNetParams.nNumLayers-CONFIG_strParams.nNumTrain_N_UpperLayerEpochs}, NM_strNetParams.mClassWeights);
                                end
                                if (CONFIG_strParams.bDynamicPenaltyBarrier == 1)
                                    alpha = alpha*lambda;
                                end

                                % Do vector to matrix conversion
                                nOffset = 0;
                                
                                % Net weights
                                for layer = ((NM_strNetParams.nNumLayers-NM_strNetParams.nNumLayers-CONFIG_strParams.nNumTrain_N_UpperLayerEpochs)+1) : NM_strNetParams.nNumLayers
                                    NM_strNetParams.cWeights{layer} = reshape(X(nOffset + 1 : nOffset + size(NM_strNetParams.cWeights{layer}(:)',2)), size(NM_strNetParams.cWeights{layer} , 1), size(NM_strNetParams.cWeights{layer} , 2));
                                    nOffset = nOffset + size(NM_strNetParams.cWeights{layer}(:)',2);
                                end
                                
                                % Class weights
                                NM_strNetParams.mClassWeights = reshape(X(nOffset + 1 : nOffset+size(NM_strNetParams.mClassWeights(:)',2)), size(NM_strNetParams.mClassWeights , 1), size(NM_strNetParams.mClassWeights , 2));

                                % Update base units input layer weights by intermediate weights
                                for tempUnit = ((NM_strNetParams.nNumLayers-NM_strNetParams.nNumLayers-CONFIG_strParams.nNumTrain_N_UpperLayerEpochs)+1) : NM_strNetParams.nNumLayers
                                    NM_strNetParams.cUnitWeights{tempUnit}{1} = NM_strNetParams.cWeights{tempUnit};
                                end

                            end
                        else
                            % Do matrix to vecotr conversion of weigths
                            % to be minimized                            
                            vWeightsToMinimize = [];
                            vLayersSizeToMinimize = [];
                            for layer = ((NM_strNetParams.nNumLayers-CONFIG_numTrainUpperLayer)+1) : NM_strNetParams.nNumLayers
                                vWeightsToMinimize = [vWeightsToMinimize NM_strNetParams.cWeights{layer}(:)'];
                                vLayersSizeToMinimize = [vLayersSizeToMinimize; vLayersSize(layer)];
                            end
                            vWeightsToMinimize = [vWeightsToMinimize NM_strNetParams.mClassWeights(:)']';
                            vLayersSizeToMinimize = [vLayersSizeToMinimize; vLayersSize(NM_strNetParams.nNumLayers+1); vLayersSize(NM_strNetParams.nNumLayers+2)];

                            % Start the minimizer
                            [X, fX] = minimize(vWeightsToMinimize, 'CG_CLASSIFY_N_Layers', CONFIG_strParams.nMaxIterCGMinimizer, vLayersSizeToMinimize,...
                                                mTopLayerActivations{NM_strNetParams.nNumLayers-CONFIG_strParams.nNumTrain_N_UpperLayerEpochs}, mCurrTrainMiniBatchTargets,...
                                                CONFIG_strParams.eMappingMode, NM_strNetParams.cUnitWeights{NM_strNetParams.nNumLayers-CONFIG_numTrainUpperLayer},...
                                                NM_strNetParams.cWeights{NM_strNetParams.nNumLayers-CONFIG_numTrainUpperLayer}, NM_strNetParams.mClassWeights);
                            
                            % Update weights by vector to matrix conversion
                            nOffset = 0;
                            
                            % Net weights
                            for layer = ((NM_strNetParams.nNumLayers-CONFIG_numTrainUpperLayer)+1) : NM_strNetParams.nNumLayers
                                NM_strNetParams.cWeights{layer} = reshape(X(nOffset + 1 : nOffset + size(NM_strNetParams.cWeights{layer}(:)',2)), size(NM_strNetParams.cWeights{layer} , 1), size(NM_strNetParams.cWeights{layer} , 2));
                                nOffset = nOffset + size(NM_strNetParams.cWeights{layer}(:)',2);
                            end
                            
                            % Update class weights by vector to matrix conversion
                            NM_strNetParams.mClassWeights = reshape(X(nOffset + 1 : nOffset+size(NM_strNetParams.mClassWeights(:)',2)), size(NM_strNetParams.mClassWeights , 1), size(NM_strNetParams.mClassWeights , 2));

                            % Update base units input layer weights by intermediate weights
                            for tempUnit = ((NM_strNetParams.nNumLayers-CONFIG_numTrainUpperLayer)+1) : NM_strNetParams.nNumLayers
                                NM_strNetParams.cUnitWeights{tempUnit}{1} = NM_strNetParams.cWeights{tempUnit};
                            end
                        end
                    else
                        if CONFIG_strParams.bBPEnablePenaltyIter == 1
                            for i = 1 : CONFIG_strParams.nNumPenaltyIterations
                                % Do matrix to vecotr conversion of weigths
                                % to be minimized                                
                                vWeightsToMinimize = [];
                                for(layer = 1 : NM_strNetParams.nNumLayers)
                                    vWeightsToMinimize = [vWeightsToMinimize NM_strNetParams.cWeights{layer}(:)'];
                                end
                                vWeightsToMinimize = [vWeightsToMinimize NM_strNetParams.mClassWeights(:)'];
                                vWeightsToMinimize = vWeightsToMinimize';

                                vLayersSizeToMinimize = [];
                                for(layer = 1 : (NM_strNetParams.nNumLayers+2))
                                    vLayersSizeToMinimize = [vLayersSizeToMinimize; vLayersSize(layer)];
                                end

                                % Start the minimizer
                                alpha  = CONFIG_strParams.nSquareBarrierMinimizerAlpha;
                                lambda = CONFIG_strParams.nDynMinimizerLambda;

                                switch (CONFIG_strParams.eMappingMode)
                                    case 'DEPTH_BASE_UNIT_MAPPING'
                                        [X, fX] = minimize(vWeightsToMinimize, 'CG_CLASSIFY_CONSTRAINED', CONFIG_strParams.nMaxIterCGMinimizer ,vLayersSizeToMinimize,...
                                                           mCurrTrainMiniBatchData, mCurrTrainMiniBatchTargets, NM_strNetParams.cBaseUnitWeights, mCurrTrainMiniBatchData,...
                                                           mPrevClassWeights, alpha, CONFIG_strParams.eMappingMode, NM_strNetParams.cUnitWeights, NM_strNetParams.cWeights,...
                                                           NM_strNetParams.mClassWeights);
                                    otherwise
                                        [X, fX] = minimize(vWeightsToMinimize, 'CG_CLASSIFY_CONSTRAINED', CONFIG_strParams.nMaxIterCGMinimizer,vLayersSizeToMinimize,...
                                                           mCurrTrainMiniBatchData, mCurrTrainMiniBatchTargets, cPrevWeights, mCurrTrainMiniBatchData,...
                                                           mPrevClassWeights, alpha, CONFIG_strParams.eMappingMode, NM_strNetParams.cUnitWeights, NM_strNetParams.cWeights,...
                                                           NM_strNetParams.mClassWeights);
                                end % end siwtch

                                if (CONFIG_strParams.nDynMinimizerLambda == 1)
                                    alpha = alpha*lambda;
                                end

                                % Update weights by vector to matrix conversion
                                nOffset = 0;
                                
                                % Net weights
                                for layer = 1 : NM_strNetParams.nNumLayers
                                    NM_strNetParams.cWeights{layer} = reshape(X(nOffset + 1 : nOffset + size(NM_strNetParams.cWeights{layer}(:)',2)), size(NM_strNetParams.cWeights{layer} , 1), size(NM_strNetParams.cWeights{layer} , 2));
                                    nOffset = nOffset + size(NM_strNetParams.cWeights{layer}(:)',2);
                                end
                                
                                % Class weights
                                NM_strNetParams.mClassWeights = reshape(X(nOffset + 1 : nOffset+size(NM_strNetParams.mClassWeights(:)',2)), size(NM_strNetParams.mClassWeights , 1), size(NM_strNetParams.mClassWeights , 2));

                                % Update base units input layer weights by intermediate weights
                                for tempUnit = 1 : size(NM_strNetParams.cWeights ,2)
                                    NM_unitWeights{tempUnit}{1} = NM_strNetParams.cWeights{tempUnit};
                                end
                            end
                        else
                            % Do matrix to vecotr conversion of weigths
                            % to be minimized                             
                            vWeightsToMinimize = [];
                            for(layer = 1 : NM_strNetParams.nNumLayers)
                                vWeightsToMinimize = [vWeightsToMinimize NM_strNetParams.cWeights{layer}(:)'];
                            end
                            vWeightsToMinimize = [vWeightsToMinimize NM_strNetParams.mClassWeights(:)'];
                            vWeightsToMinimize = vWeightsToMinimize';

                            vLayersSizeToMinimize = [];
                            for(layer = 1 : (NM_strNetParams.nNumLayers+2))
                                vLayersSizeToMinimize = [vLayersSizeToMinimize; vLayersSize(layer)];
                            end

                            % Start the minimizer
                            if CONFIG_strParams.bDepthMapping == 1 & CONFIG_depthCascadedDataRepresentation == 1
                              [XX tempDataAugmented] = NM_baseUnitActivation(mCurrTrainMiniBatchData, BP_baseUnitWeights);
                            else
                              %XX = [mCurrTrainMiniBatchData ones(nNumExamplesPerMiniBatch,1)];
                            end

                            [X, fX] = minimize(vWeightsToMinimize, 'CG_CLASSIFY', CONFIG_strParams.nMaxIterCGMinimizer, vLayersSizeToMinimize,...
                                               mCurrTrainMiniBatchData, mCurrTrainMiniBatchTargets, CONFIG_strParams.eMappingMode, NM_strNetParams.cUnitWeights,...
                                               NM_strNetParams.cWeights, NM_strNetParams.mClassWeights);

                            % Update weights by vector to matrix conversion
                            nOffset = 0;
                            
                            % Net weights
                            for layer = 1 : NM_strNetParams.nNumLayers
                                NM_strNetParams.cWeights{layer} = reshape(X(nOffset + 1 : nOffset + size(NM_strNetParams.cWeights{layer}(:)',2)), size(NM_strNetParams.cWeights{layer} , 1), size(NM_strNetParams.cWeights{layer} , 2));
                                nOffset = nOffset + size(NM_strNetParams.cWeights{layer}(:)',2);
                            end
                            
                            % Class weights
                            NM_strNetParams.mClassWeights = reshape(X(nOffset + 1 : nOffset+size(NM_strNetParams.mClassWeights(:)',2)), size(NM_strNetParams.mClassWeights , 1), size(NM_strNetParams.mClassWeights , 2));

                            % Update base units input layer weights by intermediate weights
                            for tempUnit = 1 : size(NM_strNetParams.cWeights ,2)
                                NM_unitWeights{tempUnit}{1} = NM_strNetParams.cWeights{tempUnit};
                            end
                        end
                    end
                end 
            end
    %%%%%%%%%%%%%%% END OF CONJUGATE GRADIENT WITH 3 LINESEARCHES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        end % end of minibatches loop
     
     
end % end function