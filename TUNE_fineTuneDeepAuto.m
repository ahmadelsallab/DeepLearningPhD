% Function:
% - Makes initialization of the top class layer
% - Runs backpropagation
% Inputs:
% NM_strNetParams: The net parameters to be tuned
% CONFIG_strParams: The configurations
% TST_strPerformanceInfo: The structure to update test results in it
% mTrainBatchTargets, mTrainBatchData, mTestBatchTargets, mTestBatchData: see BM_makeBatches
% nPhase: The current phase of classifier re-use (mapping)
% nNumPhases: Total number of phases for classifier re-use (mapping)
% hFidLog: Handle of the log file
% bMapping: Is classifier re-use enabled
% nBitfieldLength: The bitfield length of the Raw features
% vChunkLength: Vector storing the boundaries of each features chunk
% vOffset: The bitfield boundaries corresponding to each chunk
% eFeaturesMode: Raw, Normal, Binary, Bitfield
% Output:
% None
function [NM_strNetParams, TST_strPerformanceInfo] = TUNE_fineTuneDeepAuto(NM_strNetParams, CONFIG_strParams, TST_strPerformanceInfo,...
                                                                                mTrainBatchTargets, mTrainBatchData, mTestBatchTargets, mTestBatchData,...
                                                                                nPhase, nNumPhases, hFidLog, bMapping,...
                                                                                nBitfieldLength, vChunkLength, vOffset, eFeaturesMode)

    CLS_strPrvt.nBPMaxEpoch = CONFIG_strParams.nBPNumEpochsDuringMapping;
    fprintf(1,'\nFine-tuning deep autoencoder by minimizing cross entropy error. \n');
    
    % The number of targets is the 2nd dimension of the 3-D matrix mTrainBatchTargets
    CLS_strPrvt.nNumTargets = size(mTrainBatchTargets, 2);
    
    CLS_strPrvt.cPrevWeights = NM_strNetParams.cWeights;
    CLS_strPrvt.mPrevClassWeights = [];
        
    %%%% PREINITIALIZE WEIGHTS OF THE DISCRIMINATIVE MODEL%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%        
% l1=size(w1,1)-1;
% l2=size(w2,1)-1;
% l3=size(w3,1)-1;
% l4=size(w4,1)-1;
% l5=size(w5,1)-1;
% l6=size(w6,1)-1;
% l7=size(w7,1)-1;
% l8=size(w8,1)-1;
% l9=l1; 
    for(nLayer = 1 : NM_strNetParams.nNumLayers * 2) % nNumLayers is execluding the input data and target layers
        CLS_strPrvt.vLayersSize(nLayer) = size(NM_strNetParams.cWeights{nLayer}, 1) - 1; % we have to remove the bias (-1)
    end
    CLS_strPrvt.vLayersSize(nLayer + 1) = CLS_strPrvt.vLayersSize(1);

    TST_strPerformanceInfo.vTestErr=[];
    TST_strPerformanceInfo.vTrainErr=[];
    % Initialize test error to huge number
    TST_strPerformanceInfo.nMinTestError = 1000000;

    %%%%%%%%%% END OF PREINITIALIZATION OF WEIGHTS  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%%%%%%%%%%%%%%%% START FINE TUNING  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for nEpoch = 1 : CLS_strPrvt.nBPMaxEpoch
        
    %%%%%%%%%%%%%%%%%%%% COMPUTE TRAINING RECONSTRUCTION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        [TST_strPerformanceInfo.vTrainErr(nEpoch)] =...
            TST_computeReconstructionErr(mTrainBatchData, NM_strNetParams, nBitfieldLength, vChunkLength, vOffset, eFeaturesMode);
    %%%%%%%%%%%%%% END OF COMPUTING TRAINING MISCLASSIFICATION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%%%%%%%%%%%% COMPUTE TEST MISCLASSIFICATION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        [TST_strPerformanceInfo.vTestErr(nEpoch)] =...
            TST_computeReconstructionErr(mTestBatchData, NM_strNetParams, nBitfieldLength, vChunkLength, vOffset, eFeaturesMode);        
    %%%%%%%%%%%%%%% END OF COMPUTING TEST MISCLASSIFICATION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
        [nNumTrainExamples nNumTrainFeatures nNumTrainBatches]=size(mTrainBatchData);
        [nNumTestExamples nNumTestFeatures nNumTestBatches]=size(mTestBatchData);

        if(strcmp(eFeaturesMode, 'Raw'))
            nNumTrainFeatures = nBitfieldLength;
            nNumTestFeatures = nBitfieldLength;
        end

        % Log the result to display and log file
        fprintf(hFidLog,'Before epoch %d Train # misclassified: %d (from %d). Test # misclassified: %d (from %d) \t \t \n',...
                nEpoch, TST_strPerformanceInfo.vTrainErr(nEpoch), nNumTrainExamples * nNumTrainBatches, TST_strPerformanceInfo.vTestErr(nEpoch),nNumTestExamples * nNumTestBatches);

        fprintf(1,'Before epoch %d Train # misclassified: %d (from %d). Test # misclassified: %d (from %d) \t \t \n',...
                nEpoch, TST_strPerformanceInfo.vTrainErr(nEpoch), nNumTrainExamples * nNumTrainBatches, TST_strPerformanceInfo.vTestErr(nEpoch),nNumTestExamples * nNumTestBatches);

        % Keep the weigths if error is minimized
        if TST_strPerformanceInfo.vTestErr(nEpoch) <= TST_strPerformanceInfo.nMinTestError
            TST_strPerformanceInfo.nMinTestError = TST_strPerformanceInfo.vTestErr(nEpoch);
            for(ctrLayer = 1 : NM_strNetParams.nNumLayers * 2)
                CLS_strPrvt.cMinErrWeights{ctrLayer} = NM_strNetParams.cWeights{ctrLayer};
            end
        end
    %%%%%%%%%%%%%%%%%%%% START BACKPROP %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%     
        [cWeights, NM_strNetParams] = BP_startReconstructionBackProp(CONFIG_strParams, NM_strNetParams, mTrainBatchData, mTrainBatchTargets,...
                                                       nEpoch, CLS_strPrvt.vLayersSize, CLS_strPrvt.cPrevWeights, CLS_strPrvt.mPrevClassWeights, nPhase, nNumPhases, bMapping,...
                                                       nBitfieldLength, vChunkLength, vOffset, eFeaturesMode);
    %%%%%%%%%%%%%%%%%%%% END BACKPROP %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

         % If bBPKeepMinWeightsEveryEpoch is configured, then save the
         % weigths giving min. error each epoch
         if CONFIG_strParams.bBPKeepMinWeightsEveryEpoch == 1
            for ctrLayer = 1 : NM_strNetParams.nNumLayers * 2
                NM_strNetParams.cWeights{ctrLayer} = CLS_strPrvt.cMinErrWeights{ctrLayer};
            end
         end
    end % end of epoches loop
    
    % Keep weights giving min. error over all epochs
    for ctrLayer = 1 : NM_strNetParams.nNumLayers * 2
        NM_strNetParams.cWeights{ctrLayer} = CLS_strPrvt.cMinErrWeights{ctrLayer};
    end

end % end function