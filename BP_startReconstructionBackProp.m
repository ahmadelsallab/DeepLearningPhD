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
function [cWeights, NM_strNetParams] = BP_startReconstructionBackProp(CONFIG_strParams, NM_strNetParams, mTrainBatchData, mTrainBatchTargets,...
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
            % Matrix to vector operation for both weights and
            % layers sizes to be minimized over
%             w1=[vishid; hidrecbiases];
%             w2=[hidpen; penrecbiases];
%             w3=[hidpen2; penrecbiases2];
%             w4=[hidtop; toprecbiases];
%             w5=[hidtop'; topgenbiases]; 
%             w6=[hidpen2'; hidgenbiases2]; 
%             w7=[hidpen'; hidgenbiases]; 
%             w8=[vishid'; visbiases];            
            vWeightsToMinimize = [];
            for(layer = 1 : NM_strNetParams.nNumLayers * 2)
                vWeightsToMinimize = [vWeightsToMinimize NM_strNetParams.cWeights{layer}(:)'];
            end
            vWeightsToMinimize = vWeightsToMinimize';

%             l1=size(w1,1)-1;
%             l2=size(w2,1)-1;
%             l3=size(w3,1)-1;
%             l4=size(w4,1)-1;
%             l5=size(w5,1)-1;
%             l6=size(w6,1)-1;
%             l7=size(w7,1)-1;
%             l8=size(w8,1)-1;
%             l9=l1;
            vLayersSizeToMinimize = [];
            for(layer = 1 : NM_strNetParams.nNumLayers * 2)
                vLayersSizeToMinimize = [vLayersSizeToMinimize; vLayersSize(layer)];
            end
            vLayersSizeToMinimize = [vLayersSizeToMinimize; vLayersSize(1)];

            % Start the minimizer    
            [X, fX] = minimize(vWeightsToMinimize,'CG_RECONSTRUCT',CONFIG_strParams.nMaxIterCGMinimizer,vLayersSizeToMinimize,mCurrTrainMiniBatchData);

            % Update class weights
            % Vector to matrix operation to reverse the
            % operation before minimizer call

            nOffset = 0;

            % Net weights
            for(layer = 1 : NM_strNetParams.nNumLayers * 2)
                NM_strNetParams.cWeights{layer} = reshape(X(nOffset+1:nOffset+(vLayersSize(layer)+1) * vLayersSize(layer+1)), vLayersSize(layer)+1, vLayersSize(layer+1));
                nOffset = nOffset + (vLayersSize(layer)+1)*vLayersSize(layer+1);
            end                               

    %%%%%%%%%%%%%%% END OF CONJUGATE GRADIENT WITH 3 LINESEARCHES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        end % end of minibatches loop
     
     
end % end function