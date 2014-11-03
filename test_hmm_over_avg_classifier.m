clear, clc;
load input_data;
load final_net;
[nErr, vTargetsDNN] = TST_computeClassificationErrDNN(mTrainFeatures, mTrainTargets, NM_strNetParams, 0, 0, 0, 1, 1, 'ABSOLUTE_ERR_CALC');
vTargetsDNN = vTargetsDNN';
[I J]=max(mTrainTargets, [], 2);
vTargets = J';
nStates = 8;
contextLen = size(vTargets, 2);
% [mHMMTransitionProbs_, mHMMEmissionProbs_] = hmmestimate(vTargetsDNN, vTargets, 'PSEUDOEMISSIONS', rand(nStates, nStates), 'PSEUDOTRANSITIONS', rand(nStates, nStates));
% % Viterbi is fast
% [mHMMTransitionProbs, mHMMEmissionProbs] = hmmtrain(vTargetsDNN, mHMMTransitionProbs_, mHMMEmissionProbs_, 'VERBOSE', true, 'ALGORITHM', 'Viterbi');
% %[mHMMTransitionProbs, mHMMEmissionProbs] = hmmtrain(vTargetsDNN, mHMMTransitionProbs_, mHMMEmissionProbs_, 'VERBOSE', true);
% vTargetsHMM = hmmviterbi(vTargetsDNN, mHMMTransitionProbs, mHMMEmissionProbs);
% nTrainErr = size(find(vTargetsHMM ~= vTargets), 2)
% % Train another layer of HMM
% [mHMMTransitionProbs, mHMMEmissionProbs] = hmmtrain(vTargetsHMM, mHMMTransitionProbs, mHMMEmissionProbs, 'VERBOSE', true, 'ALGORITHM', 'Viterbi');
% vTargetsHMM = hmmviterbi(vTargetsHMM, mHMMTransitionProbs, mHMMEmissionProbs);
% 
% nTrainErr = size(find(vTargetsHMM ~= vTargets), 2)
% 
% [mHMMTransitionProbs, mHMMEmissionProbs] = hmmtrain(vTargetsHMM, mHMMTransitionProbs, mHMMEmissionProbs, 'VERBOSE', true, 'ALGORITHM', 'Viterbi');
% vTargetsHMM = hmmviterbi(vTargetsHMM, mHMMTransitionProbs, mHMMEmissionProbs);
% 
% nTrainErr = size(find(vTargetsHMM ~= vTargets), 2)
% 
% 
% [mHMMTransitionProbs, mHMMEmissionProbs] = hmmtrain(vTargetsHMM, mHMMTransitionProbs, mHMMEmissionProbs, 'VERBOSE', true, 'ALGORITHM', 'Viterbi');
% vTargetsHMM = hmmviterbi(vTargetsHMM, mHMMTransitionProbs, mHMMEmissionProbs);
% 
% nTrainErr = size(find(vTargetsHMM ~= vTargets), 2)
% 
% 
% % Test err
% [nErr, vTargetsDNN] = TST_computeClassificationErrDNN(mTestFeatures, mTestTargets, NM_strNetParams, 0, 0, 0, 1, 1, 'ABSOLUTE_ERR_CALC');
% vTargetsDNN = vTargetsDNN';
% [I J]=max(mTestTargets, [], 2);
% vTargets = J';
% vTargetsHMM = hmmviterbi(vTargetsDNN, mHMMTransitionProbs, mHMMEmissionProbs);
% nTestErr = size(find(vTargetsHMM ~= vTargets), 2)
% 
% % [mHMMTransitionProbs_, mHMMEmissionProbs_] = hmmestimate(vTargetsDNN(1:contextLen), vTargets(1:contextLen), 'PSEUDOEMISSIONS', rand(nStates, nStates), 'PSEUDOTRANSITIONS', rand(nStates, nStates));
% % %[mHMMTransitionProbs, mHMMEmissionProbs] = hmmtrain(vTargetsDNN(1:contextLen), mHMMTransitionProbs_, mHMMEmissionProbs_, 'VERBOSE', true, 'PSEUDOEMISSIONS', rand(nStates, nStates), 'PSEUDOTRANSITIONS', rand(nStates, nStates), 'ALGORITHM', 'Viterbi');
% % [mHMMTransitionProbs, mHMMEmissionProbs] = hmmtrain(vTargetsDNN(1:contextLen), mHMMTransitionProbs_, mHMMEmissionProbs_, 'VERBOSE', true);
% % vTargetsHMM = hmmviterbi(vTargetsDNN(1:contextLen), mHMMTransitionProbs, mHMMEmissionProbs);
% % nTrainErr = size(find(vTargetsHMM ~= vTargets(1:contextLen)), 2);


nAccTrainErr = 0;
nExamples = size(vTargets, 2);
N = floor(nExamples/contextLen);
if(mod(nExamples, contextLen) == 0)
    N_HMM = N;
    sizeHMM = contextLen * ones(1, N_HMM);% sizes of all pieces is equal to context length
else
    N_HMM = N+1;
    sizeHMM = contextLen * ones(1, N_HMM);% sizes of all pieces is equal to context length
    sizeHMM(N_HMM) = nExamples - N*contextLen;% size of last piece is the remaining
end

startIdx = 1;
fprintf(1,'\n');
i = 1;
vTargetsHMM = [];
while(i <= N_HMM)
    fprintf(1,'Context %d\n', i);
    endIdx = startIdx + sizeHMM(i) - 1;
    [mHMMTransitionProbs_, mHMMEmissionProbs_] = hmmestimate(vTargetsDNN(startIdx:endIdx), vTargets(startIdx:endIdx), 'PSEUDOEMISSIONS', rand(nStates, nStates), 'PSEUDOTRANSITIONS', rand(nStates, nStates));    
    %[mHMMTransitionProbs, mHMMEmissionProbs] = hmmtrain(vTargetsDNN(startIdx:endIdx), mHMMTransitionProbs_, mHMMEmissionProbs_, 'VERBOSE', true);
    % Viterbi is fast
    [mHMMTransitionProbs, mHMMEmissionProbs] = hmmtrain(vTargetsDNN(startIdx:endIdx), mHMMTransitionProbs_, mHMMEmissionProbs_, 'VERBOSE', true, 'ALGORITHM', 'Viterbi');
    vTargetsHMM_ = hmmviterbi(vTargetsDNN(startIdx:endIdx), mHMMTransitionProbs, mHMMEmissionProbs);
    vTargetsHMM = [vTargetsHMM vTargetsHMM_];
    nTrainErr = size(find(vTargetsHMM_ ~= vTargets(startIdx:endIdx)), 2)
    nAccTrainErr = nAccTrainErr + nTrainErr  
    startIdx = startIdx + sizeHMM(i);
    i = i + 1;

end

nTrainErrFinal = size(find(vTargetsHMM ~= vTargets), 2)

%%%%%% TEST ERROR %%%%%
[nErr, vTargetsDNN] = TST_computeClassificationErrDNN(mTestFeatures, mTestTargets, NM_strNetParams, 0, 0, 0, 1, 1, 'ABSOLUTE_ERR_CALC');
[I J]=max(mTestTargets, [], 2);
vTargets = J';
nAccTrainErr = 0;
nExamples = size(vTargets, 2);
N = floor(nExamples/contextLen);
if(mod(nExamples, contextLen) == 0)
    N_HMM = N;
    sizeHMM = contextLen * ones(1, N_HMM);% sizes of all pieces is equal to context length
else
    N_HMM = N+1;
    sizeHMM = contextLen * ones(1, N_HMM);% sizes of all pieces is equal to context length
    sizeHMM(N_HMM) = nExamples - N*contextLen;% size of last piece is the remaining
end

startIdx = 1;
fprintf(1,'\n');
i = 1;
vTargetsHMM = [];
while(i <= N_HMM)
    fprintf(1,'Context %d\n', i);
    endIdx = startIdx + sizeHMM(i) - 1;
    %[mHMMTransitionProbs_, mHMMEmissionProbs_] = hmmestimate(vTargetsDNN(startIdx:endIdx), vTargets(startIdx:endIdx), 'PSEUDOEMISSIONS', rand(nStates, nStates), 'PSEUDOTRANSITIONS', rand(nStates, nStates));    
    %[mHMMTransitionProbs, mHMMEmissionProbs] = hmmtrain(vTargetsDNN(startIdx:endIdx), mHMMTransitionProbs_, mHMMEmissionProbs_, 'VERBOSE', true);
    % Viterbi is fast
    %[mHMMTransitionProbs, mHMMEmissionProbs] = hmmtrain(vTargetsDNN(startIdx:endIdx), mHMMTransitionProbs_, mHMMEmissionProbs_, 'VERBOSE', true, 'ALGORITHM', 'Viterbi');
    vTargetsHMM_ = hmmviterbi(vTargetsDNN(startIdx:endIdx), mHMMTransitionProbs, mHMMEmissionProbs);
    vTargetsHMM = [vTargetsHMM vTargetsHMM_];
    nTrainErr = size(find(vTargetsHMM_ ~= vTargets(startIdx:endIdx)), 2)
    nAccTrainErr = nAccTrainErr + nTrainErr  
    startIdx = startIdx + sizeHMM(i);
    i = i + 1;

end

nTestErrFinal = size(find(vTargetsHMM ~= vTargets), 2)
