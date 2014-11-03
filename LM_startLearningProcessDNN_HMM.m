% Function:
% It starts the learning process, including Pre-trainig and initialization, fine tuning and results logging.
% Initialization includes pre-training if needed.
% Inputs:
% CONFIG_strParams: The configuration parameters
% mTrainTargets, mTrainFeatures, mTestTargets, mTestFeatures: Used to build
% confusion matrix
% mTrainBatchTargets, mTrainBatchData, mTestBatchTargets, mTestBatchData: The train and test batch data and targets
% mTrainFeatures, mTrainTargets, mTestTargets: Used to train and test the
% HMM on input data
% nBitfieldLength: The length of converted bitfield
% vChunkLength: Vector storing the boundaries of each features chunk
% vOffset: The bitfield boundaries corresponding to each chunk
% Output:
% None
function LM_startLearningProcessDNN_HMM(CONFIG_strParams,...
                                        mTrainFeatures, mTrainTargets, mTestFeatures, mTestTargets,...
                                        mTrainBatchTargets, mTrainBatchData, mTestBatchTargets, mTestBatchData, nBitfieldLength, vChunkLength, vOffset)

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
        
        if(strcmp(CONFIG_strParams.eFeaturesMode, 'Raw'))
            nFeaturesVecLen = nBitfieldLength;
        else
            nFeaturesVecLen = size(mTrainBatchData, 2);
        end
        
        % Initiaize Net
        [NM_strNetParams] =...
            NM_initializeNet(LM_strLearningPrcocessPrvt.bMapping, CONFIG_strParams.eMappingMode, CONFIG_strParams.eMappingDirection, CONFIG_strParams.bDepthCascadedDataRepMode,...
                             NM_strNetParams, LM_strLearningPrcocessPrvt.nPhase, LM_strLearningPrcocessPrvt.hFidLog,...
                             nFeaturesVecLen, PRE_strPrvt.nMaxEpoch, mTrainBatchData, vChunkLength, vOffset, CONFIG_strParams.eFeaturesMode);

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
	
    fprintf(LM_strLearningPrcocessPrvt.hFidLog,'Start training HMM...\n');
    fprintf(1,'Start training HMM...\n');

    % Train HMM    
    [HMM_strParams.mHMMTransitionProbs, HMM_strParams.mHMMEmissionProbs] = CLS_trainHMMOnTopDNN(mTrainFeatures, mTrainTargets, NM_strNetParams, CONFIG_strParams,...
                                                                                                LM_strLearningPrcocessPrvt.bMapping,...
                                                                                                LM_strLearningPrcocessPrvt.nPhase, LM_strLearningPrcocessPrvt.nNumPhases);
    fprintf(LM_strLearningPrcocessPrvt.hFidLog,'Finished training HMM\n');
    fprintf(1,'Finished training HMM\n');

    fprintf(LM_strLearningPrcocessPrvt.hFidLog,'Start testing HMM\n');
    fprintf(1,'Start testing HMM\n');

    
    % Compute train and test errors
    [TST_strPerformanceInfo.nHMMTrainErr, TST_strPerformanceInfo.nHMMTestErr, vTrainTargetsHMM, vTestTargetsHMM] =...
        TST_computeClassificationErrDNN_HMM(mTrainFeatures, mTrainTargets, mTestFeatures, mTestTargets, NM_strNetParams, HMM_strParams, CONFIG_strParams);

    fprintf(LM_strLearningPrcocessPrvt.hFidLog,'HMM Test Error %d (out of %d)\n', TST_strPerformanceInfo.nHMMTestErr, size(mTestFeatures, 1));
    fprintf(1,'HMM Test Error %d (out of %d)\n', TST_strPerformanceInfo.nHMMTestErr, size(mTrainFeatures, 1));

    fprintf(LM_strLearningPrcocessPrvt.hFidLog,'HMM Train Error %d (out of %d)\n', TST_strPerformanceInfo.nHMMTrainErr, size(mTrainFeatures, 1));
    fprintf(1,'HMM Train Error %d (out of %d)\n', TST_strPerformanceInfo.nHMMTrainErr, size(mTrainFeatures, 1));

    
    fprintf(LM_strLearningPrcocessPrvt.hFidLog,'Finished testing HMM\n');
    fprintf(1,'Finished testing HMM\n');

    % Save the HMM params
    save(CONFIG_strParams.sHMMWorkSpaceFileName, 'HMM_strParams');
    
    % Build confusion matrix
    if(CONFIG_strParams.bBuildConfusionMatrix == 1)
        
        %%%%%%%%%%%%%% TRAIN CONFUSION MATRIX %%%%%%%%%%%%%%%%%%%%%%%%
        fprintf(1,'Start Train Building Confusion Matrix...\n');
        fprintf(LM_strLearningPrcocessPrvt.hFidLog, 'Start Train Building Confusion Matrix...\n');
        % Get the train target vector
        [I vTrainTargets]=max(mTrainTargets, [], 2);

        % Obtain the output train targets
        vTrainTargetsOut = map(MAXENT_clsParams, mTrainFeatures);

        % Build confusion matrix
        [TST_strPerformanceInfo.mTrainConfusionMatrix, TST_strPerformanceInfo.mTrainNormalConfusionMatrix, TST_strPerformanceInfo.vTrainNumTrainExamplesPerClass,...
        TST_strPerformanceInfo.vTrainAccuracyPerClass, TST_strPerformanceInfo.nTrainOverallAccuracy] = LM_buildConfusionMatrix(vTrainTargets', vTrainTargetsOut);

        fprintf(1,'End Train Building Confusion Matrix\n');
        fprintf(LM_strLearningPrcocessPrvt.hFidLog, 'End Train Building Confusion Matrix\n');

        %%%%%%%%%%%%%% TEST CONFUSION MATRIX %%%%%%%%%%%%%%%%%%%%%%%%%
        fprintf(1,'Start Test Building Confusion Matrix...\n');
        fprintf(LM_strLearningPrcocessPrvt.hFidLog, 'Start Test Building Confusion Matrix...\n');
        % Get the train target vector
        [I vTestTargets]=max(mTestTargets, [], 2);

        % Obtain the output train targets
        vTestTargetsOut = map(MAXENT_clsParams, mTestFeatures);

        % Build confusion matrix
        [TST_strPerformanceInfo.mTestConfusionMatrix, TST_strPerformanceInfo.mTestNormalConfusionMatrix, TST_strPerformanceInfo.vTestNumTrainExamplesPerClass,...
        TST_strPerformanceInfo.vTestAccuracyPerClass, TST_strPerformanceInfo.nTestOverallAccuracy] = LM_buildConfusionMatrix(vTestTargets', vTestTargetsOut);

        fprintf(1,'End Test Building Confusion Matrix\n');
        fprintf(LM_strLearningPrcocessPrvt.hFidLog, 'End Test Building Confusion Matrix\n');
        
    end % end if    
    
    % Save the current configuration in the error performance workspace
    save(CONFIG_strParams.sNameofErrWorkspace, 'TST_strPerformanceInfo');
    
	% Close the log file
    fclose(LM_strLearningPrcocessPrvt.hFidLog);
	
end % end function