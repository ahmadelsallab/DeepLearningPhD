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
function LM_startLearningProcessDNN(CONFIG_strParams,...
                                    mTestFeatures, mTestTargets, mTrainFeatures, mTrainTargets,...
                                    mTrainBatchTargets, mTrainBatchData, mTestBatchTargets, mTestBatchData, nBitfieldLength, vChunkLength, vOffset, mAutoLabels)

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
		case 'SAME'
            LM_strLearningPrcocessPrvt.nNumPhases = CONFIG_strParams.nDesiredAdaptivePhases - 1;			
        case 'NONE'
            LM_strLearningPrcocessPrvt.nNumPhases = LM_strLearningPrcocessPrvt.nPhase; 
            NM_strNetParams.nNumLayers = CONFIG_strParams.nInitialNumLayers;
        otherwise
            fprintf(1,'Wrong configuration of mapping direction');
    end % end switch

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
			case 'SAME'
			    NM_strNetParams.vLayersWidths = CONFIG_strParams.vInitialLayersWidths;
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
                             nFeaturesVecLen, PRE_strPrvt.nMaxEpoch, mTrainBatchData, nBitfieldLength, vChunkLength, vOffset, CONFIG_strParams.eFeaturesMode);

        fprintf(LM_strLearningPrcocessPrvt.hFidLog,'Initializing done successfuly\n');
        fprintf(1,'Initializing done successfuly\n');
        
        fprintf(LM_strLearningPrcocessPrvt.hFidLog,'Fine tuning ...\n');
        fprintf(1,'Fine tuning ...\n');


        % if(CONFIG_strParams.bReduceTrainingSetSizeWithMapping == 0 & CONFIG_strParams.nDesiredTrainSetSizePercent > 0)
            % fprintf(1, 'Reducing training set to %d percent...\n', CONFIG_strParams.nDesiredTrainSetSizePercent);


            % mTrainFeatures = mTrainFeatures(1 : size(mTrainFeatures, 1) * (CONFIG_strParams.nDesiredTrainSetSizePercent / 100), :);
            % mTrainTargets = mTrainTargets(1 : size(mTrainTargets, 1) * (CONFIG_strParams.nDesiredTrainSetSizePercent / 100), :);

            % [mTrainTargets, mAutoLabels] = DPREP_autoLabel(mTrainTargets, CONFIG_strParams);
            % [mTrainBatchTargets, mTrainBatchData] =... 
                % BM_makeBatches(mTestFeatures,...
                               % mTestTargets,...
                               % mTrainFeatures,...
                               % mTrainTargets,...
                               % CONFIG_strParams.nMaxFeaturesRange,...
                               % CONFIG_strParams.bAutoLabel,...
                               % mAutoLabels,...
                               % CONFIG_strParams.nBatchSize);

            % fprintf(1, 'Reduction done successfuly\n');

        % end
        % Fine tune and classify
        [NM_strNetParams, TST_strPerformanceInfo] = CLS_fineTuneAndClassifyDNN(NM_strNetParams, CONFIG_strParams, TST_strPerformanceInfo,...
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
        
        
        if (LM_strLearningPrcocessPrvt.bMapping == 0)
            % Latch mapping
            LM_strLearningPrcocessPrvt.bMapping = 1;
            
            % Save basic unit weights
            NM_strNetParams.cBaseUnitWeights = NM_strNetParams.cWeights;
            
            if(strcmp(CONFIG_strParams.eMappingDirection, 'DEPTH') &&...
               strcmp(CONFIG_strParams.eMappingMode, 'DEPTH_CASCADED_DATA_REPRESENTATION_MAPPING'))
               
                NM_strNetParams.cCascadedBaseUnitWeights = NM_strNetParams.cBaseUnitWeights;
                
            end
			
			% Double training size
			if(CONFIG_strParams.bDoubleTrainingSetSizeWithMapping == 1)
													
					nReducedSize = size(mTrainFeatures, 1) * 2^(LM_strLearningPrcocessPrvt.nPhase + 1);
                    load(CONFIG_strParams.sInputDataWorkspace, 'mTrainFeatures', 'mTrainTargets');
					mTrainFeatures = mTrainFeatures(1 : nReducedSize, :);
					mTrainTargets = mTrainTargets(1 : nReducedSize, :);
					save('input_data_reduced');			
								

					% Load auto label data if enabled
					[mTrainTargets, mAutoLabels] = DPREP_autoLabel(mTrainTargets, CONFIG_strParams);
				 

					fprintf(1, 'Making train and test batches...\n');
					% Make train and test batches
					[mTrainBatchTargets, mTrainBatchData, mTestBatchTargets, mTestBatchData] =... 
						BM_makeBatches(mTestFeatures,...
									   mTestTargets,...
									   mTrainFeatures,...
									   mTrainTargets,...
									   CONFIG_strParams.nMaxFeaturesRange,...
									   CONFIG_strParams.bAutoLabel,...
									   mAutoLabels,...
									   CONFIG_strParams.nBatchSize);
					fprintf(1, 'Batches made successfuly\n');
            else
                if(CONFIG_strParams.bReduceTrainingSetSizeWithMapping == 1 && CONFIG_strParams.nDesiredTrainSetSizePercent > 0)
                    fprintf(1, 'Reducing training set to %d percent...\n', CONFIG_strParams.nDesiredTrainSetSizePercent);

                    load(CONFIG_strParams.sInputDataWorkspace, 'mTrainFeatures', 'mTrainTargets');
                    mTrainFeatures = mTrainFeatures(1 : size(mTrainFeatures, 1) * (CONFIG_strParams.nDesiredTrainSetSizePercent / 100), :);
                    mTrainTargets = mTrainTargets(1 : size(mTrainTargets, 1) * (CONFIG_strParams.nDesiredTrainSetSizePercent / 100), :);

                    [mTrainTargets, mAutoLabels] = DPREP_autoLabel(mTrainTargets, CONFIG_strParams);
                    [mTrainBatchTargets, mTrainBatchData] =... 
                        BM_makeBatches(mTestFeatures,...
                                       mTestTargets,...
                                       mTrainFeatures,...
                                       mTrainTargets,...
                                       CONFIG_strParams.nMaxFeaturesRange,...
                                       CONFIG_strParams.bAutoLabel,...
                                       mAutoLabels,...
                                       CONFIG_strParams.nBatchSize);

                    fprintf(1, 'Reduction done successfuly\n');

                end
			end

            
        end
        
        % Advance to next learning phase
        LM_strLearningPrcocessPrvt.nPhase = LM_strLearningPrcocessPrvt.nPhase + 1;
        
        % Get the minimum error of this phase
        TST_strPerformanceInfo.vTestErrPerPhase(LM_strLearningPrcocessPrvt.nPhase) = min(TST_strPerformanceInfo.vTestErr);
        
        % Log it to the log file
        fprintf(LM_strLearningPrcocessPrvt.hFidLog, 'Classification error of phase %d: %d\n', LM_strLearningPrcocessPrvt.nPhase, min(TST_strPerformanceInfo.vTestErr));
        fprintf(1, 'Classification error of phase %d: %d\n', LM_strLearningPrcocessPrvt.nPhase, min(TST_strPerformanceInfo.vTestErr));

        save(CONFIG_strParams.sNetDataWorkspace, 'NM_strNetParams');
        
    end % end-while: DNN training phases

	% Plot the error performance over the whole learning process
    plot(TST_strPerformanceInfo.vTestErrPerPhase);
	
    % Save the error performance info
    save(CONFIG_strParams.sNetDataWorkspace, 'NM_strNetParams');
    
    % Build confusion matrix
    if(CONFIG_strParams.bBuildConfusionMatrix == 1)
        
        %%%%%%%%%%%%%% TRAIN CONFUSION MATRIX %%%%%%%%%%%%%%%%%%%%%%%%
        fprintf(1,'Start Train Building Confusion Matrix...\n');
        fprintf(LM_strLearningPrcocessPrvt.hFidLog, 'Start Train Building Confusion Matrix...\n');
       
        % Get the train target vector
        [I vTrainTargets]=max(mTrainTargets, [], 2);
        
        % Get the DNN output
        [nErr, vObtainedTrainTargets, vDesiredTrainTargets] = TST_computeClassificationErrDNN(mTrainBatchData, mTrainBatchTargets, NM_strNetParams, 0, 0, 0, 1, 1, 'ABSOLUTE_ERR_CALC',...
                                                                   nBitfieldLength, vChunkLength, vOffset, CONFIG_strParams.eFeaturesMode);

        % Build confusion matrix
        [TST_strPerformanceInfo.mTrainConfusionMatrix, TST_strPerformanceInfo.mTrainNormalConfusionMatrix, TST_strPerformanceInfo.vTrainNumTrainExamplesPerClass,...
        TST_strPerformanceInfo.vTrainAccuracyPerClass, TST_strPerformanceInfo.nTrainOverallAccuracy] = LM_buildConfusionMatrix(vDesiredTrainTargets, vObtainedTrainTargets);

        fprintf(1,'End Train Building Confusion Matrix\n');
        fprintf(LM_strLearningPrcocessPrvt.hFidLog, 'End Train Building Confusion Matrix\n');

        %%%%%%%%%%%%%% TEST CONFUSION MATRIX %%%%%%%%%%%%%%%%%%%%%%%%%
        fprintf(1,'Start Test Building Confusion Matrix...\n');
        fprintf(LM_strLearningPrcocessPrvt.hFidLog, 'Start Test Building Confusion Matrix...\n');
       
        % Get the train target vector
        [I vTestTargets]=max(mTestTargets, [], 2);
        
        % Get the DNN output
        [nErr, vObtainedTestTargets, vDesiredTestTargets] = TST_computeClassificationErrDNN(mTestBatchData, mTestBatchTargets, NM_strNetParams, 0, 0, 0, 1, 1, 'ABSOLUTE_ERR_CALC',...
                                                                  nBitfieldLength, vChunkLength, vOffset, CONFIG_strParams.eFeaturesMode);

        % Build confusion matrix
        [TST_strPerformanceInfo.mTestConfusionMatrix, TST_strPerformanceInfo.mTestNormalConfusionMatrix, TST_strPerformanceInfo.vTestNumTrainExamplesPerClass,...
        TST_strPerformanceInfo.vTestAccuracyPerClass, TST_strPerformanceInfo.nTestOverallAccuracy] = LM_buildConfusionMatrix(vDesiredTestTargets, vObtainedTestTargets);

        fprintf(1,'End Test Building Confusion Matrix\n');
        fprintf(LM_strLearningPrcocessPrvt.hFidLog, 'End Test Building Confusion Matrix\n');
        
    end % end if 
       
	% Save the current configuration in the error performance workspace
    save(CONFIG_strParams.sNameofErrWorkspace, 'TST_strPerformanceInfo');
	
	% Close the log file
    fclose(LM_strLearningPrcocessPrvt.hFidLog);
	
end % end function