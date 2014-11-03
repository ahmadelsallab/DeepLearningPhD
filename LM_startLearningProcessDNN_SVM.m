% Function:
% It fine tunes the target layer probs with SVM. This is different from SVM
% on top of DNN since the later takes the last layer activations, while
% here we take the target layer probs (Soft-max output) as inputs to SVM
% Inputs:
% CONFIG_strParams: The configuration parameters
% mTrainFeatures, mTrainTargets, mTestTargets: Used to train and test the
% HMM on input data
% Output:
% None
function LM_startLearningProcessDNN_SVM(CONFIG_strParams,...
                                        mTrainFeatures, mTrainTargets, mTestFeatures, mTestTargets)


    
    % Start private data
    LM_strLearningPrcocessPrvt.bMapping = 0;
    LM_strLearningPrcocessPrvt.nPhase = 0;
    LM_strLearningPrcocessPrvt.hFidLog = fopen(CONFIG_strParams.sLearnLogFile,'w');
    
    % Initialize Network
    NM_strNetParams.cWeights = [];
    NM_strNetParams.cPrevWeights = [];
    NM_strNetParams.cBaseUnitWeights = [];
    NM_strNetParams.cUnitWeights = [];
    NM_strNetParams.cCascadedBaseUnitWeights = [];
    NM_strNetParams.mClassWeights = [];
    NM_strNetParams.mPrevClassWeights = [];
    NM_strNetParams.nNumLayers = 0;
    NM_strNetParams.nPrevNumLayers = 0;
    
    % Initialize TST structure
    TST_strPerformanceInfo = [];

    switch(CONFIG_strParams.eMappingDirection)
        case 'DEPTH'
            switch(CONFIG_strParams.eMappingMode)
                case 'NN_DEPTH_CLASSIFIER_MAPPING'
                    LM_strLearningPrcocessPrvt.nNumPhases = CONFIG_strParams.nFinalNumLayers/CONFIG_strParams.nInitialNumLayers - 1;
                    NM_strNetParams.nNumLayers = CONFIG_strParams.nInitialNumLayers;
                case 'DEPTH_CASCADED_DATA_REPRESENTATION_MAPPING'
                    LM_strLearningPrcocessPrvt.nNumPhases = CONFIG_finalNumLayers/CONFIG_strParams.nInitialNumLayers - 1;
                    NM_strNetParams.nNumLayers = CONFIG_strParams.nInitialNumLayers;
                case 'DEPTH_BASE_UNIT_MAPPING'
                    LM_strLearningPrcocessPrvt.nNumPhases = CONFIG_strParams.nDepthBaseUnitMappingNumberOfStackedUnits - 1;
                    NM_strNetParams.nNumLayers = CONFIG_strParams.nInitialNumLayers;
                otherwise
                    % Do nothing
            end % end switch
        case 'BREADTH'
            LM_strLearningPrcocessPrvt.nNumPhases = floor(CONFIG_strParams.nFinalFirstLayerWidth/CONFIG_strParams.vInitialLayersWidths(1));
            LM_strLearningPrcocessPrvt.nNumPhases = log(LM_strLearningPrcocessPrvt.nNumPhases)/log(2);
        case 'NONE'
            LM_strLearningPrcocessPrvt.nNumPhases = LM_strLearningPrcocessPrvt.nPhase; 
            NM_strNetParams.nNumLayers = CONFIG_strParams.nInitialNumLayers;
        otherwise
            fprintf(1,'Wrong configuration of mapping direction');
    end % end switch

    fprintf(LM_strLearningPrcocessPrvt.hFidLog,'Start training DNN...\n');
    fprintf(1,'Start training DNN...\n');
    
    % Start learning cycles
    while (LM_strLearningPrcocessPrvt.nPhase <= LM_strLearningPrcocessPrvt.nNumPhases)
        
        % Pretraining is permitted only (if enabled) before mapping pahses occur
        if ((LM_strLearningPrcocessPrvt.bMapping == 0) && (CONFIG_strParams.bEnablePretraining == 1))
            PRE_strPrvt.nMaxEpoch = CONFIG_strParams.nPreTrainEpochs; 
        else
            PRE_strPrvt.nMaxEpoch = 0; 
        end % end-if
        switch(CONFIG_strParams.eMappingDirection)
            case 'DEPTH'
                NM_strNetParams.nPrevNumLayers = NM_strNetParams.nNumLayers;
                NM_strNetParams.vLayersWidths = CONFIG_strParams.vInitialLayersWidths;
                switch(CONFIG_strParams.eMappingMode)
                    case 'NN_DEPTH_CLASSIFIER_MAPPING'
                        NM_strNetParams.nNumLayers = CONFIG_strParams.nInitialNumLayers * 2^LM_strLearningPrcocessPrvt.nPhase;
                        NM_strNetParams.vLayersWidths = [NM_strNetParams.vLayersWidths NM_strNetParams.vLayersWidths];
                    case 'DEPTH_CASCADED_DATA_REPRESENTATION_MAPPING'
                        % Do nothing
                    case 'DEPTH_BASE_UNIT_MAPPING'
                        % Do nothing
                    otherwise
                        % Do nothing
                end % end switch
            case 'BREADTH'
                NM_strNetParams.vLayersWidths = CONFIG_strParams.vInitialLayersWidths * 2^LM_strLearningPrcocessPrvt.nPhase;
                NM_strNetParams.nNumLayers = CONFIG_strParams.nInitialNumLayers;       
            case 'NONE'
            otherwise
                fprintf(1,'Wrong configuration of mapping direction');
        end % end switch
        
        fprintf(LM_strLearningPrcocessPrvt.hFidLog,'Training sub-net with arch ');
        fprintf(1,'Training sub-net with arch ');
        
        for j = 1 : CONFIG_strParams.nInitialNumLayers
            fprintf(LM_strLearningPrcocessPrvt.hFidLog, '%d ', NM_strNetParams.vLayersWidths(j));
            fprintf(1, '%d ', NM_strNetParams.vLayersWidths(j));
        end % end-for
        
        fprintf(LM_strLearningPrcocessPrvt.hFidLog,'\n');
        fprintf(1,'\n'); 

        
        fprintf(1,'Initializing net...\n');
        fprintf(LM_strLearningPrcocessPrvt.hFidLog,'Initializing net...\n');
        % Initiaize Net
        [NM_strNetParams] =...
            NM_initializeNet(LM_strLearningPrcocessPrvt.bMapping, CONFIG_strParams.eMappingMode, CONFIG_strParams.eMappingDirection, CONFIG_strParams.bDepthCascadedDataRepMode,...
                             NM_strNetParams, LM_strLearningPrcocessPrvt.nPhase, LM_strLearningPrcocessPrvt.hFidLog,...
                             size(mTrainBatchData, 2), PRE_strPrvt.nMaxEpoch, mTrainBatchData);

        fprintf(LM_strLearningPrcocessPrvt.hFidLog,'Initializing done successfuly\n');
        fprintf(1,'Initializing done successfuly\n');
        
        fprintf(LM_strLearningPrcocessPrvt.hFidLog,'Fine tuning ...\n');
        fprintf(1,'Fine tuning ...\n');

        % Fine tune and classify
        [NM_strNetParams, TST_strPerformanceInfo] = CLS_fineTuneAndClassifyDNN(NM_strNetParams, CONFIG_strParams, TST_strPerformanceInfo,...
                                                                            mTrainBatchTargets, mTrainBatchData, mTestBatchTargets, mTestBatchData,...
                                                                            LM_strLearningPrcocessPrvt.nPhase, LM_strLearningPrcocessPrvt.nNumPhases,...
                                                                            LM_strLearningPrcocessPrvt.hFidLog, LM_strLearningPrcocessPrvt.bMapping);
        fprintf(LM_strLearningPrcocessPrvt.hFidLog,'Fine tuning done successfuly\n');
        fprintf(1, 'Fine tuning done successfuly\n');
        
        fprintf(LM_strLearningPrcocessPrvt.hFidLog,'Update Net\n');
        fprintf(1, 'Update Net\n');
        
        if (LM_strLearningPrcocessPrvt.bMapping == 1 &&...
			strcmp(CONFIG_strParams.eMappingDirection, 'DEPTH') &&...
			strcmp(CONFIG_strParams.eMappingMode, 'DEPTH_CASCADED_DATA_REPRESENTATION_MAPPING'))
			
            NM_strNetParams.cCascadedBaseUnitWeights = [NM_strNetParams.cCascadedBaseUnitWeights NM_strNetParams.cWeights];
			
        end % end-if
        
        
        if (LM_strLearningPrcocessPrvt.bMapping == 0)
            % Latch mapping
            LM_strLearningPrcocessPrvt.bMapping = 1;
            
            % Save basic unit weights
            NM_strNetParams.cBaseUnitWeights = NM_strNetParams.cWeights;
            
            if(strcmp(CONFIG_strParams.eMappingDirection, 'DEPTH') &&...
               strcmp(CONFIG_strParams.eMappingMode, 'DEPTH_CASCADED_DATA_REPRESENTATION_MAPPING'))
               
                NM_strNetParams.cCascadedBaseUnitWeights = NM_strNetParams.cBaseUnitWeights;
                
            end
            
        end
        
        % Advance to next learning phase
        LM_strLearningPrcocessPrvt.nPhase = LM_strLearningPrcocessPrvt.nPhase + 1;
        
        % Get the minimum error of this phase
        TST_strPerformanceInfo.vTestErrPerPhase(LM_strLearningPrcocessPrvt.nPhase) = min(TST_strPerformanceInfo.vTestErr);
        
        % Log it to the log file
        fprintf(LM_strLearningPrcocessPrvt.hFidLog, 'Classification error of phase %d: %d\n', LM_strLearningPrcocessPrvt.nPhase, min(TST_strPerformanceInfo.vTestErr));
        fprintf(1, 'Classification error of phase %d: %d\n', LM_strLearningPrcocessPrvt.nPhase, min(TST_strPerformanceInfo.vTestErr));

    end % end-while: DNN training phases

    fprintf(LM_strLearningPrcocessPrvt.hFidLog,'Finished training DNN\n');
    fprintf(1,'Finished training DNN\n');
	% Plot the error performance over the whole learning process
    plot(TST_strPerformanceInfo.vTestErrPerPhase);
	
    % Save the error performance info
    save(CONFIG_strParams.sNetDataWorkspace, 'NM_strNetParams');       
	
    fprintf(LM_strLearningPrcocessPrvt.hFidLog,'Start training and testing SVM...\n');
    fprintf(1,'Start training and testing SVM...\n');
    
    % Feed the train features to obtain target layer probs
    [nErr, vTargetOut, mTrainSVMFeatures] = NM_feedNet(mTrainFeatures, mTrainTargets, NM_strNetParams, 0, 0, 0, 1, 1, 'ABSOLUTE_ERR_CALC');
    [nErr, vTargetOut, mTestSVMFeatures] = NM_feedNet(mTestFeatures, mTestTargets, NM_strNetParams, 0, 0, 0, 1, 1, 'ABSOLUTE_ERR_CALC');

    % Train and test SVM
    CLS_trainAndTestSVM(CONFIG_strParams, TST_strPerformanceInfo, LM_strLearningPrcocessPrvt.hFidLog, mTestSVMFeatures, mTestTargets, mTrainSVMFeatures, mTrainTargets);

    fprintf(LM_strLearningPrcocessPrvt.hFidLog,'Finished training and testing SVM\n');
    fprintf(1,'Finished training and testing SVM\n');

    
	% Close the log file
    fclose(LM_strLearningPrcocessPrvt.hFidLog);
	
end % end function