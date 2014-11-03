clear, clc;

% Compute A: transision matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load input_data;
[I J]=max(mTrainTargets, [], 2);
%[I J]=max(mTestTargets, [], 2);
vTargets = J;
nNumTargets = size(mTrainTargets, 2);
%nNumTargets = size(mTestTargets, 2);

% Initialize A
A = zeros(nNumTargets, nNumTargets);

% Loop on all targets and fill-in A
% Loop only before the last target example, because no transition FROM the last
% Also, no transition TO the first, so start from 2
for i = 2 : size(vTargets, 1) - 1 
    % Increment the transition from vTargets(i-1) to vTargets(i)
    A(vTargets(i-1), vTargets(i)) = A(vTargets(i-1), vTargets(i)) + 1;
end

% Normalize A
vSumColOfA = sum(A, 2);
mDivMatrix = repmat(vSumColOfA, 1, nNumTargets);
A = A ./ mDivMatrix;
%A = diag([1 1 1 1 1 1 1 1]);
% Override A with the one of hmm
%[I J]=max(mTrainTargets, [], 2);
% [I J]=max(mTestTargets, [], 2);
% vTargets = J';
% nStates = 8;
% contextLen = size(vTargets, 2);
% [A, mHMMEmissionProbs_] = hmmestimate(vTargets, vTargets, 'PSEUDOEMISSIONS', rand(nStates, nStates), 'PSEUDOTRANSITIONS', rand(nStates, nStates));
% load final_net;
% %[nErr, vTargetsDNN] = TST_computeClassificationErrDNN(mTrainBatchData, mTrainBatchTargets, NM_strNetParams, 0, 0, 0, 1, 1,  'ABSOLUTE_ERR_CALC', nBitfieldLength, vChunkLength, vOffset, 'Raw');
% %[A, mHMMEmissionProbs] = hmmtrain(vTargetsDNN, A, mHMMEmissionProbs_, 'VERBOSE', true);
% [A, mHMMEmissionProbs] = hmmtrain(vTargets, A, mHMMEmissionProbs_, 'VERBOSE', true, 'ALGORITHM', 'Viterbi');

%%%%%% TRAIN ERROR %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%

% % Loop on all input data and perform N-gram Viterbi decoding
load final_net;
contextLen = 3;%size(vTargets, 2);
% 
% nAccTrainErr = 0;
% nExamples = size(vTargets, 1);
% N = floor(nExamples/contextLen);
% if(mod(nExamples, contextLen) == 0)
%     Ncontexts = N;
%     nContextExamples = contextLen * ones(1, Ncontexts);% sizes of all pieces is equal to context length
% else
%     Ncontexts = N+1;
%     nContextExamples = contextLen * ones(1, Ncontexts);% sizes of all pieces is equal to context length
%     nContextExamples(Ncontexts) = nExamples - N*contextLen;% size of last piece is the remaining
% end
% 
% startIdx = 1;
% fprintf(1,'TRAIN ERROR\n');
% i = 1;
% vTargetsCRF = [];
% while(i <= Ncontexts)
%     fprintf(1,'Context %d\n', i);
%     endIdx = startIdx + nContextExamples(i) - 1;
%     
%     % Get the upper layer probs = emission of the context examples-->Feed
%     % fwd DNN
%     [mUpperProbs] = TST_computeUpperLayerProbsDNN(mTrainFeatures(startIdx:endIdx, :), nNumTargets, NM_strNetParams, nBitfieldLength, vChunkLength, vOffset, 'Raw');
%     
%     % The emission matrix
%     B = mUpperProbs';
%     
%     % The sequqnce is just the indices of colomns of B = rows of mUpperProbs
%     SEQ = linspace(1, nContextExamples(i), nContextExamples(i));
%     
%     vTargetsCRF_ = hmmviterbi(SEQ, A, B);
%     vTargetsCRF = [vTargetsCRF vTargetsCRF_];
%     nTrainErr = size(find(vTargetsCRF_ ~= vTargets(startIdx:endIdx)'), 2)
%     nAccTrainErr = nAccTrainErr + nTrainErr  
%     startIdx = startIdx + nContextExamples(i);
%     i = i + 1;
% 
% end
% 
% nTrainErrFinal = size(find(vTargetsCRF ~= vTargets), 2)
% nTrainErrFinal = size(find(vTargetsCRF ~= vTargets), 2) / size(vTargets, 1)

%%%%%% TEST ERROR %%%%%
%%%%%%%%%%%%%%%%%%%%%%%
[I J]=max(mTestTargets, [], 2);
vTargets = J;

% Loop on all input data and perform N-gram Viterbi decoding
nAccTestErr = 0;
nExamples = size(vTargets, 1);
N = floor(nExamples/contextLen);
if(mod(nExamples, contextLen) == 0)
    Ncontexts = N;
    nContextExamples = contextLen * ones(1, Ncontexts);% sizes of all pieces is equal to context length
else
    Ncontexts = N+1;
    nContextExamples = contextLen * ones(1, Ncontexts);% sizes of all pieces is equal to context length
    nContextExamples(Ncontexts) = nExamples - N*contextLen;% size of last piece is the remaining
end

startIdx = 1;
fprintf(1,'TEST ERROR\n');
i = 1;
vTargetsCRF = [];
while(i <= Ncontexts)
    fprintf(1,'Context %d\n', i);
    endIdx = startIdx + nContextExamples(i) - 1;
    
    % Get the upper layer probs = emission of the context examples-->Feed
    % fwd DNN
    [mUpperProbs] = TST_computeUpperLayerProbsDNN(mTestFeatures(startIdx:endIdx, :), nNumTargets, NM_strNetParams, nBitfieldLength, vChunkLength, vOffset, 'Raw');
    
    % The emission matrix
    B = mUpperProbs';
    
    % The sequqnce is just the indices of colomns of B = rows of mUpperProbs
    SEQ = linspace(1, nContextExamples(i), nContextExamples(i));
    
    vTargetsCRF_ = hmmviterbi(SEQ, A, B);
    vTargetsCRF = [vTargetsCRF vTargetsCRF_];
    nTestErr = size(find(vTargetsCRF_ ~= vTargets(startIdx:endIdx)'), 2)
    nAccTestErr = nAccTestErr + nTestErr  
    startIdx = startIdx + nContextExamples(i);
    i = i + 1;

end

nTestErrFinalCRF = size(find(vTargetsCRF ~= vTargets'), 2)
nTestErrFinalCRF = size(find(vTargetsCRF ~= vTargets'), 2) / size(vTargets, 1)
nAccuracyFinalCRF = (1 - nTestErrFinalCRF)

clear;
load input_data mTestBatchData mTestBatchTargets nBitfieldLength vChunkLength vOffset;
load final_net NM_strNetParams;
[nTestErrFinalDNN, vTargetsDNN] = TST_computeClassificationErrDNN(mTestBatchData, mTestBatchTargets, NM_strNetParams, 0, 0, 0, 1, 1,  'ABSOLUTE_ERR_CALC', nBitfieldLength, vChunkLength, vOffset, 'Raw');
nTestErrFinalDNN
nTestErrFinalDNN = nTestErrFinalDNN / (size(vTargetsDNN, 1)*size(vTargetsDNN, 2))
nAccuracyFinalDNN = (1 - nTestErrFinalDNN)
save err_performance;