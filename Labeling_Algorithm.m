% Load auto encoder weights
load(CONFIG_strParams.sNetDataWorkspace);

% Load train features and desired labels
 fprintf(1, 'Converting input files...\n');
 
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
      load(CONFIG_strParams.sInputDataWorkspace, 'nBitfieldLength', 'vChunkLength', 'vOffset');
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
      load(CONFIG_strParams.sInputDataWorkspace, 'mTestFeatures', 'mTestTargets', 'mTrainFeatures', 'mTrainTargets');
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

code_dim = size(NM_strNetParams.cWeights{NM_strNetParams.nNumLayers}, 2);

nNumTargets = size(DPREP_strData.mTrainTargets, 2);

numfeatures = size(DPREP_strData.mTrainFeatures, 1);

% Initialize codes
codes = [];
  
% Initialize clusters means
means = zeros(nNumTargets, code_dim);

% Loop on all train features
nExamplesPerClass = zeros(nNumTargets, 1);
[nNumExamples nNumFeatures]=size(DPREP_strData.mTrainFeatures);
for i = 1 : nNumExamples
  
  % The actual of the feature is the position that has 1 in the desired target
  labels(i) = find(DPREP_strData.mTrainTargets(i,:)==1);
  
  %Calculate code: Feed forward

  % Normal activation sequence, feeding non-augmented data
  % Feed Fwd        
  % Convert to normal format of data
  if(strcmp(CONFIG_strParams.eFeaturesMode, 'Raw'))
    [mCurrData] = DCONV_convertRawToBitfield(DPREP_strData.mTrainFeatures(i, :), DPREP_strData.nBitfieldLength, DPREP_strData.vChunkLength, DPREP_strData.vOffset);
  else
     mCurrData = DPREP_strData.mTrainFeatures(i, :);
  end
  [mTempNotAugActivationData mAugActivationData] = NM_neuralNetActivation(mCurrData, NM_strNetParams.cWeights(1 : NM_strNetParams.nNumLayers - 1));
  %w4probs = w3probs*w4; w4probs = [w4probs  ones(N,1)]; NO NEED TO AUGMENT
  mAugActivationData{NM_strNetParams.nNumLayers} = mAugActivationData{NM_strNetParams.nNumLayers - 1} * NM_strNetParams.cWeights{NM_strNetParams.nNumLayers};
  %code = [mAugActivationData{NM_strNetParams.nNumLayers} ones(1, 1)];
  code = mAugActivationData{NM_strNetParams.nNumLayers};
  
  %Append to the means of the label of the current feature
  means(labels(i),:) = means(labels(i),:) + code; %or mean(code) returns row contains average of every colomn
  nExamplesPerClass(labels(i)) = nExamplesPerClass(labels(i)) + 1;
  % Append the codes
  % codes: totnumcases X 30 = sum(curr_cluster_size) X 30 
  codes = [codes; code];
  
  fprintf(1,'training example = %d \n', i);
 end
 
%Calculate means of each cluster to inintialize k-means algorithm 
for i = 1 : nNumTargets
	means(i,:) = means(i,:)/nExamplesPerClass(i);
end
 
 % Apply K-means
 %[IDX,C] = kmeans(codes, 10, 'distance', 'sqEuclidean', 'onlinephase', 'off');
 [new_labels, new_means] = kmeans_dev_1(codes, nNumTargets, labels, means);
 
%Calculate distances
for i = 1 : size(codes,1) % numcases
	for j = 1 : nNumTargets
		 %D(j,i) = euc_dist(codes(i,:), means(j,:));
		 D(j,i) = sum((codes(i,:) - means(j,:)).^2);
	end;
end;

 % Compare indicies and difference in means
 idx_accuracy = size(find((new_labels-labels)== 0),2)/size(labels,2)*100 %sum((new_labels-label)s.^2)/size(labels, 1);
 [M N] = size(means);
 means_err = sum(sum((new_means-means).^2)) /(M*N);
 
 %save auto_label.mat;
 save(CONFIG_strParams.sAutoLabelWorkSpaceFileName);