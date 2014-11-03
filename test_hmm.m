clear, clc;
load input_data mTrainFeatures mTrainTargets mTestFeatures mTestTargets;
vTrainFeatures = mTrainFeatures(:,4);
[I vTrainTargets]=max(mTrainTargets, [], 2);
%vTrainTargets = vTrainTargets';

vTestFeatures = mTestFeatures(:,4);
[I vTestTargets]=max(mTestTargets, [], 2);
%vTestTargets = vTestTargets';

nStates = 8;
%contextLen = size(vTrainTargets, 2);
contextLen = 11;

nAccTrainErr = 0;
nExamples = size(vTrainTargets, 1);
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
    %[mHMMTransitionProbs_, mHMMEmissionProbs_] = hmmestimate(mTrainFeatures(startIdx:endIdx), vTrainTargets(startIdx:endIdx), 'PSEUDOEMISSIONS', rand(nStates, nStates), 'PSEUDOTRANSITIONS', rand(nStates, nStates));    
    [mHMMTransitionProbs_, mHMMEmissionProbs_] = hmmestimate(vTrainFeatures(startIdx:endIdx), vTrainTargets(startIdx:endIdx));    
    %[mHMMTransitionProbs, mHMMEmissionProbs] = hmmtrain(vTargetsDNN(startIdx:endIdx), mHMMTransitionProbs_, mHMMEmissionProbs_, 'VERBOSE', true);
    % Viterbi is fast
    %[mHMMTransitionProbs, mHMMEmissionProbs] = hmmtrain(vTrainFeatures(startIdx:endIdx), mHMMTransitionProbs_, mHMMEmissionProbs_, 'VERBOSE', true, 'ALGORITHM', 'Viterbi');
    [mHMMTransitionProbs, mHMMEmissionProbs] = hmmtrain(vTrainFeatures(startIdx:endIdx), mHMMTransitionProbs_, mHMMEmissionProbs_, 'VERBOSE', false, 'ALGORITHM', 'Viterbi');
    vTargetsHMM_ = hmmviterbi(vTrainFeatures(startIdx:endIdx), mHMMTransitionProbs, mHMMEmissionProbs);
    vTargetsHMM_ = vTargetsHMM_';
    vTargetsHMM = [vTargetsHMM vTargetsHMM_];
    nTrainErr = size(find(vTargetsHMM_ ~= vTrainTargets(startIdx:endIdx)), 1)
    nAccTrainErr = nAccTrainErr + nTrainErr  
    startIdx = startIdx + sizeHMM(i);
    i = i + 1;

end

nTrainErrFinal = size(find(vTargetsHMM ~= vTargets), 1)

%%%%%% TEST ERROR %%%%%
nAccTrainErr = 0;
nExamples = size(vTargets, 1);
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
    vTargetsHMM_ = hmmviterbi(vTestFeatures(startIdx:endIdx), mHMMTransitionProbs, mHMMEmissionProbs);
    vTargetsHMM_ = vTargetsHMM_';
    vTargetsHMM = [vTargetsHMM vTargetsHMM_];
    nTestErr = size(find(vTargetsHMM_ ~= vTestTargets(startIdx:endIdx)), 1)
    nAccTestErr = nAccTestErr + nTestErr  
    startIdx = startIdx + sizeHMM(i);
    i = i + 1;

end

nTestErrFinal = size(find(vTargetsHMM ~= vTestTargets), 1)
