% Function:
% It starts the learning process, including Pre-trainig and initialization, fine tuning and results logging.
% Initialization includes pre-training if needed.
% Inputs:
% CONFIG_strParams: The configuration parameters
% mTrainTargets, mTrainFeatures, mTestTargets, mTestFeatures: Used to build
% confusion matrix
% mTrainBatchTargets, mTrainBatchData, mTestBatchTargets, mTestBatchData: The train and test batch data and targets
% nBitfieldLength: The length of converted bitfield
% vChunkLength: Vector storing the boundaries of each features chunk
% vOffset: The bitfield boundaries corresponding to each chunk
% Output:
% None
function LM_startLearningProcessDeepAuto(CONFIG_strParams,...
                                         mTestFeatures, mTestTargets, mTrainFeatures, mTrainTargets,...
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
    NM_strNetParams.vLayersWidths = CONFIG_strParams.vInitialLayersWidths;
    
    % Initialize TST structure
    TST_strPerformanceInfo = [];

    LM_strLearningPrcocessPrvt.nNumPhases = LM_strLearningPrcocessPrvt.nPhase; 
    NM_strNetParams.nNumLayers = CONFIG_strParams.nInitialNumLayers;

  
    PRE_strPrvt.nMaxEpoch = CONFIG_strParams.nPreTrainEpochs; 

    
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
        NM_initializeNetDeepAuto(LM_strLearningPrcocessPrvt.bMapping, CONFIG_strParams.eMappingMode, CONFIG_strParams.eMappingDirection, CONFIG_strParams.bDepthCascadedDataRepMode,...
                         NM_strNetParams, LM_strLearningPrcocessPrvt.nPhase, LM_strLearningPrcocessPrvt.hFidLog,...
                         nFeaturesVecLen, PRE_strPrvt.nMaxEpoch, mTrainBatchData, nBitfieldLength, vChunkLength, vOffset, CONFIG_strParams.eFeaturesMode);

    fprintf(LM_strLearningPrcocessPrvt.hFidLog,'Initializing done successfuly\n');
    fprintf(1,'Initializing done successfuly\n');

    fprintf(LM_strLearningPrcocessPrvt.hFidLog,'Fine tuning ...\n');
    fprintf(1,'Fine tuning ...\n');

    % Fine tune and classify
    [NM_strNetParams, TST_strPerformanceInfo] = TUNE_fineTuneDeepAuto(NM_strNetParams, CONFIG_strParams, TST_strPerformanceInfo,...
                                                                      mTrainBatchTargets, mTrainBatchData, mTestBatchTargets, mTestBatchData,...
                                                                      LM_strLearningPrcocessPrvt.nPhase, LM_strLearningPrcocessPrvt.nNumPhases,...
                                                                      LM_strLearningPrcocessPrvt.hFidLog, LM_strLearningPrcocessPrvt.bMapping,...
                                                                      nBitfieldLength, vChunkLength, vOffset, CONFIG_strParams.eFeaturesMode);
    fprintf(LM_strLearningPrcocessPrvt.hFidLog,'Fine tuning done successfuly\n');
    fprintf(1, 'Fine tuning done successfuly\n');

    fprintf(LM_strLearningPrcocessPrvt.hFidLog,'Update Net\n');
    fprintf(1, 'Update Net\n');

    if (LM_strLearningPrcocessPrvt.bMapping == 1 &&...
        strcmp(CONFIG_strParams.eMappingDirection, 'DEPTH') &&...
        strcmp(CONFIG_strParams.eMappingMode, 'DEPTH_CASCADED_DATA_REPRESENTATION_MAPPING'))

        NM_strNetParams.cCascadedBaseUnitWeights = [NM_strNetParams.cCascadedBaseUnitWeights NM_strNetParams.cWeights];

    end % end-if

    % Save basic unit weights
    NM_strNetParams.cBaseUnitWeights = NM_strNetParams.cWeights;


    % Advance to next learning phase
    LM_strLearningPrcocessPrvt.nPhase = LM_strLearningPrcocessPrvt.nPhase + 1;

    % Get the minimum error of this phase
    TST_strPerformanceInfo.vTestErrPerPhase(LM_strLearningPrcocessPrvt.nPhase) = min(TST_strPerformanceInfo.vTestErr);

    % Log it to the log file
    fprintf(LM_strLearningPrcocessPrvt.hFidLog, 'Classification error of phase %d: %d\n', LM_strLearningPrcocessPrvt.nPhase, min(TST_strPerformanceInfo.vTestErr));
    fprintf(1, 'Classification error of phase %d: %d\n', LM_strLearningPrcocessPrvt.nPhase, min(TST_strPerformanceInfo.vTestErr));

    save(CONFIG_strParams.sNetDataWorkspace, 'NM_strNetParams', '-v7.3')

	% Plot the error performance over the whole learning process
    plot(TST_strPerformanceInfo.vTestErrPerPhase);
    
	% Save the current configuration in the error performance workspace
    save(CONFIG_strParams.sNameofErrWorkspace, 'TST_strPerformanceInfo');
	
	% Close the log file
    fclose(LM_strLearningPrcocessPrvt.hFidLog);
	
end % end function