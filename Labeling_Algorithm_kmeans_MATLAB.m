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
%[new_labels, new_means] = kmeans_dev_1(codes, nNumTargets, labels, means);
[new_labels_, new_means] = kmeans(codes, nNumTargets, 'distance', 'sqEuclidean', 'onlinephase', 'off', 'Start', means);


%Calculate distances
%for i = 1 : size(codes,1) % numcases
for j = 1 : nNumTargets %size(means,1) = k
		for j = 1 : nNumTargets %size(means,1) = k
			 %D(j,i) = euc_dist(codes(i,:), means(j,:));
			 %D(j,i) = sum((codes(i,:) - new_means(j,:)).^2);
			 D(j,i) = sum((means(i,:) - new_means(j,:)).^2);
		end;
end;
[min_dist , mean_examples_indices] = min(D,[],2);

% Apply original label of the mean example to the whole cluster
% 1. Get correct label (the label of the nearest example to the mean
%	[C1 C2 C3]mean_examples_indices = nearest to the mean Li
%L1				|C2 idx=1
%L2				|C1 idx=0
%L3				|C3 idx=2
%--------------------	
%labels: L2 L3 L1
%correct_labels = init_labels(1) init_labels(0) init_labels(2)

for j = 1 : nNumTargets
	correct_label(j) = labels(mean_examples_indices(j)); % correct_label = initial label of the example at the index nearest to the mean
end;

%2. Correct the labels
new_labels = new_labels_;
for i = 1 : nNumTargets
	ind = [];
	ind = find(new_labels_ == i);
	for j = 1 : size(ind,2)
		new_labels(ind(j)) = correct_label(i);
	end;
	clear ind;
end;


%Calculate distances
% for i = 1 : size(codes,1) % numcases
	% for j = 1 : nNumTargets
%		 D(j,i) = euc_dist(codes(i,:), means(j,:));
		 % D(j,i) = sum((codes(i,:) - means(j,:)).^2);
	% end;
% end;

% Compare indicies and difference in means
idx_accuracy = size(find((new_labels-labels')== 0),1)/size(labels',1)*100 %sum((new_labels-label)s.^2)/size(labels, 1);
[M N] = size(means);
means_err = sum(sum((new_means-means).^2)) /(M*N);

%save auto_label.mat;
save(CONFIG_strParams.sAutoLabelWorkSpaceFileName);
means_err = sum(sum((new_means-means).^2)) /(M*N);