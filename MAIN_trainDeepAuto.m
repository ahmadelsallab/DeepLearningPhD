% Function:
% The main entry point to train and test Generic Deep ANN
% Inputs:
% sDefaultPath: String with the default path of the input txt file
% sFeaturesFileName: String of the input txt file
% Output:
% None
function MAIN_trainDeepAuto(CONFIG_strParams)    
    
    fprintf(1, 'Converting input files...\n');
	% Check the dataset used
	switch(CONFIG_strParams.sDataset)
		case 'MNIST'
			% The output of conversion is saved in CONFIG_strParams.sInputDataWorkspace
			DCONV_convertMNIST(CONFIG_strParams);

		case '20News'
			% The output of conversion is saved in CONFIG_strParams.sInputDataWorkspace
			DCONV_convert20News(CONFIG_strParams);

		case 'Ohsumed'
			% The output of conversion is saved in CONFIG_strParams.sInputDataWorkspace
			DCONV_convertOhsumed(CONFIG_strParams);
		
		case 'Reuters'
			% The output of conversion is saved in CONFIG_strParams.sInputDataWorkspace
			DCONV_convertReuters(CONFIG_strParams);			
		
		case 'Diacritization'
			% Do nothing, a C# preprocessor has run and provided the required CONFIG_strParams.sInputDataWorkspace = input_data.mat
		otherwise
			% Do nothing
	end
    
	if(CONFIG_strParams.bReduceTrainingSetSize)
		fprintf(1, 'Reducing training set to %d percent...\n', CONFIG_strParams.nDesiredTrainSetSizePercent);
        load(CONFIG_strParams.sInputDataWorkspace);

		mTrainFeatures = mTrainFeatures(1 : size(mTrainFeatures, 1) * (CONFIG_strParams.nDesiredTrainSetSizePercent / 100), :);
		mTrainTargets = mTrainTargets(1 : size(mTrainTargets, 1) * (CONFIG_strParams.nDesiredTrainSetSizePercent / 100), :);
		
		save(CONFIG_strParams.sInputDataWorkspace, '-v7.3', 'mTestFeatures', 'mTestTargets', 'mTrainFeatures', 'mTrainTargets');
		
		fprintf(1, 'Reduction done successfuly\n');
	end
	% Check the input format
    switch (CONFIG_strParams.sInputFormat)
        case 'MATLAB'
            % Convert raw data to matlab vars
            [DPREP_strData.mFeatures, DPREP_strData.mTargets] = DCONV_convertMatlabInput();
			
			% Save converted data
			save(CONFIG_strParams.sInputDataWorkspace, '-struct', 'DPREP_strData', 'mFeatures', 'mTargets');
        case 'TxtFile'
            % Convert raw data to matlab vars
            [DPREP_strData.mFeatures, DPREP_strData.mTargets, DPREP_strData.nBitfieldLength, DPREP_strData.vChunkLength, DPREP_strData.vOffset] = DCONV_convert(CONFIG_strParams.fullRawDataFileName, CONFIG_strParams.eFeaturesMode);
			% Save converted data
			save(CONFIG_strParams.sInputDataWorkspace, '-struct', 'DPREP_strData', 'mFeatures', 'mTargets');

        case 'MatlabWorkspace'
            load(CONFIG_strParams.sInputDataWorkspace);
            DPREP_strData.mTargets = mTargets;
            DPREP_strData.mFeatures = mFeatures;
            if(strcmp(CONFIG_strParams.eFeaturesMode, 'Raw'))
                DPREP_strData.nBitfieldLength = nBitfieldLength;
                DPREP_strData.vChunkLength = vChunkLength;
                DPREP_strData.vOffset = vOffset;

                clear mTargets mFeatures nBitfieldLength vChunkLength vOffset;

            else
                DPREP_strData.nBitfieldLength = 0;
                DPREP_strData.vChunkLength = [];
                DPREP_strData.vOffset = [];

                clear mTargets mFeatures;

            end
            
		case 'MatlabWorkspaceReadyTestTrainSplit'
            load(CONFIG_strParams.sInputDataWorkspace);
            if(strcmp(CONFIG_strParams.eFeaturesMode, 'Raw'))
                DPREP_strData.nBitfieldLength = nBitfieldLength;
                DPREP_strData.vChunkLength = vChunkLength;
                DPREP_strData.vOffset = vOffset;
                clear nBitfieldLength vChunkLength vOffset;
            
            else
                DPREP_strData.nBitfieldLength = 0;
                DPREP_strData.vChunkLength = [];
                DPREP_strData.vOffset = [];
                
            end
        otherwise
            % Convert raw data to matlab vars
            [DPREP_strData.mFeatures, DPREP_strData.mTargets, DPREP_strData.nBitfieldLength DPREP_strData.vChunkLength, DPREP_strData.vOffset] = DCONV_convert(CONFIG_strParams.fullRawDataFileName, CONFIG_strParams.eFeaturesMode);          
    end

    
    fprintf(1, 'Conversion done successfuly\n');
    
	fprintf(1, 'Splitting dataset into train and test sets...\n');
	
	switch (CONFIG_strParams.sInputFormat)
		case 'MatlabWorkspaceReadyTestTrainSplit'
			%load(CONFIG_strParams.sInputDataWorkspace);
			DPREP_strData.mTestFeatures = mTestFeatures;
			DPREP_strData.mTestTargets = mTestTargets;
			DPREP_strData.mTrainFeatures = mTrainFeatures;
			DPREP_strData.mTrainTargets = mTrainTargets;
            clear mTestFeatures mTestTargets mTrainFeatures mTrainTargets;
        otherwise
			% Split into train and test sets
			[DPREP_strData.mTestFeatures, DPREP_strData.mTestTargets, DPREP_strData.mTrainFeatures, DPREP_strData.mTrainTargets] =... 
				TTS_formTrainTestSets(DPREP_strData.mFeatures,...
									  DPREP_strData.mTargets,...
									  CONFIG_strParams.sSplitCriteria,...
									  CONFIG_strParams.nTrainToTestFactor);
                
                %Save split data
                save(CONFIG_strParams.sInputDataWorkspace, '-struct', 'DPREP_strData', '-append',...
                     'mTestFeatures',...
                     'mTestTargets',...
                     'mTrainFeatures',...
                     'mTrainTargets');
                
                if(strcmp(CONFIG_strParams.sMemorySavingMode, 'ON'))
                    % clear DPREP_strData.mFeatures DPREP_strData.mTargets;
                    DPREP_strData.mFeatures = [];
                    DPREP_strData.mTargets = [];
                end
         
    end
    
	fprintf(1, 'Splitting done successfuly\n');

	fprintf(1, 'Start learning process\n');
    fprintf(1, 'Making train and test batches...\n');
    % Make train and test batches
    DPREP_strData.mAutoLabels = [];
    [DPREP_strData.mTrainBatchTargets, DPREP_strData.mTrainBatchData, DPREP_strData.mTestBatchTargets, DPREP_strData.mTestBatchData] =... 
        BM_makeBatches(DPREP_strData.mTestFeatures,...
                       DPREP_strData.mTestTargets,...
                       DPREP_strData.mTrainFeatures,...
                       DPREP_strData.mTrainTargets,...
                       CONFIG_strParams.nMaxFeaturesRange,...
                       CONFIG_strParams.bAutoLabel,...
                       DPREP_strData.mAutoLabels,...
                       CONFIG_strParams.nBatchSize);
    fprintf(1, 'Batches made successfuly\n');

    % Save Batch data
    save(CONFIG_strParams.sInputDataWorkspace, '-struct', 'DPREP_strData', '-append',... 
        'mTrainBatchTargets',... 
        'mTrainBatchData',...
        'mTestBatchTargets',...
        'mTestBatchData');

    if(strcmp(CONFIG_strParams.sMemorySavingMode, 'ON'))

        DPREP_strData.mTestFeatures = [];
        DPREP_strData.mTestTargets = [];
        DPREP_strData.mTrainFeatures = [];
        DPREP_strData.mTrainTargets = [];
        DPREP_strData.mAutoLabels = [];
    end

% DPREP_strData.mTrainBatchTargets = mTrainBatchTargets;
% clear mTrainBatchTargets;
% DPREP_strData.mTrainBatchData = mTrainBatchData;
% clear mTrainBatchData;
% DPREP_strData.mTestBatchTargets = mTestBatchTargets;
% clear mTestBatchTargets;
% DPREP_strData.mTestBatchData = mTestBatchData;
% clear mTestBatchData;

    % Start the learning process
    LM_startLearningProcessDeepAuto(CONFIG_strParams,...
                               DPREP_strData.mTestFeatures, DPREP_strData.mTestTargets, DPREP_strData.mTrainFeatures, DPREP_strData.mTrainTargets,...
                               DPREP_strData.mTrainBatchTargets, DPREP_strData.mTrainBatchData, DPREP_strData.mTestBatchTargets, DPREP_strData.mTestBatchData,...
                               DPREP_strData.nBitfieldLength, DPREP_strData.vChunkLength, DPREP_strData.vOffset);
	fprintf(1, 'Learning process performed successfuly\n'); 

end