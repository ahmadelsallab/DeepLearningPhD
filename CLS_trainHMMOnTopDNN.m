% Function:
% Trains the HMM over the target layer outputs of the DNN
% Inputs:
% mTrainFeatures: Input data. Matrix (nxm), where n is the number of examples and m is the features vector length
% mTrainTargets: Associated targets. Vector (nxl), where n is the number of examples and l is the number of target classes
% NM_strNetParams: Net parameters to feed forward
% bMapping: Is classifier mapping (re-use) enabled
% eMappingDirection: See CONFIG_setConfigParams
% eMappingMode: See CONFIG_setConfigParams
% nPhase: The current phase of classifier re-use (mapping)
% nNumPhases: Total number of phases for classifier re-use (mapping)
% eTestMode = 'EPOCH_ERR_CALC' (calculate error in each epoch) or 'ABSOLUTE_ERR_CALC' (normal err calculation) 
% Output:
% mHMMTransitionProbs: Transisition matrix of HMM
% mHMMEmissionProbs: Emission matrix of HMM
function [mHMMTransitionProbs, mHMMEmissionProbs] = CLS_trainHMMOnTopDNN(mTrainFeatures, mTrainTargets, NM_strNetParams, CONFIG_strParams, bMapping,...
                                                                         nPhase, nNumPhases)

    % Obtain the output targets of the DNN net                                        
    [nErr, vTrainTargetsDNN] = TST_computeClassificationErrDNN(mTrainFeatures, mTrainTargets, NM_strNetParams, bMapping, CONFIG_strParams.eMappingDirection,...
                                                               CONFIG_strParams.eMappingMode, nPhase, nNumPhases, 'ABSOLUTE_ERR_CALC');
    % Form the sequences (outputs of DNN)
    vTrainTargetsDNN = vTrainTargetsDNN';

    % Form the states (required outputs from HMM)
    [I J]=max(mTrainTargets, [], 2);
    vTrainTargets = J';
    
    % The number of states = number of targets
    nStates  = size(mTrainTargets, 2);

    % Train (estimate transition and emission probs) of HMM
    % Put pseudo probs to account for zero tranisition or emissions not
    % seen in training
    %[mHMMTransitionProbs, mHMMEmissionProbs] = hmmestimate(seq, states, 'PSEUDOEMISSIONS', rand(nStates, nStates), 'PSEUDOTRANSITIONS', rand(nStates, nStates));
    %[mHMMTransitionProbs, mHMMEmissionProbs] = hmmestimate(seq, states);
    
    % Do training for each context batch
    
    % Get the total number of training examples
    nExamples = size(vTrainTargets, 2);
    
    % Compute context
    if(CONFIG_strParams.nContextLength == 0)
        nContextLength = nExamples;
    else
        nContextLength = CONFIG_strParams.nContextLength;
    end
    % Initialize the training accumulator error
    nAccTrainErr = 0;
    

    
    % Get the integer number of context batches
    N = floor(nExamples/nContextLength);
    
    % Adjust according to the remaining
    if(mod(nExamples, nContextLength) == 0)
        N_HMM = N;
        sizeHMM = nContextLength * ones(1, N_HMM);% sizes of all pieces is equal to context length
    else
        N_HMM = N+1;
        sizeHMM = nContextLength * ones(1, N_HMM);% sizes of all pieces is equal to context length
        sizeHMM(N_HMM) = nExamples - N*nContextLength;% size of last piece is the remaining
    end

    startIdx = 1;
    fprintf(1,'\n');
    i = 1;
    vTargetsHMM = [];
    
    % Start training loop
    while(i <= N_HMM)
        fprintf(1,'Context %d\n', i);
        endIdx = startIdx + sizeHMM(i) - 1;
        
        % The initial targets are the outputs of DNN
        vTargetsDNN = vTrainTargetsDNN(startIdx:endIdx);
        
        % Train number of HMM layers
        for(m = 1 : CONFIG_strParams.nNumHMMLayers)
            
            % Train HMM for the current context batch
            [mHMMTransitionProbs, mHMMEmissionProbs] = hmmestimate(vTargetsDNN, vTrainTargets(startIdx:endIdx), 'PSEUDOEMISSIONS', rand(nStates, nStates), 'PSEUDOTRANSITIONS', rand(nStates, nStates));    
            %[mHMMTransitionProbs, mHMMEmissionProbs] = hmmtrain(vTrainTargetsDNN(startIdx:endIdx), mHMMTransitionProbs, mHMMEmissionProbs, 'VERBOSE', true);
            % Viterbi is fast
            %[mHMMTransitionProbs, mHMMEmissionProbs] = hmmtrain(vTargetsDNN, mHMMTransitionProbs, mHMMEmissionProbs, 'VERBOSE', false, 'ALGORITHM', 'Viterbi', 'PSEUDOEMISSIONS', rand(nStates, nStates), 'PSEUDOTRANSITIONS', rand(nStates, nStates));
            [mHMMTransitionProbs, mHMMEmissionProbs] = hmmtrain(vTargetsDNN, mHMMTransitionProbs, mHMMEmissionProbs, 'VERBOSE', false, 'ALGORITHM', 'Viterbi');%, 'PSEUDOEMISSIONS', rand(nStates, nStates), 'PSEUDOTRANSITIONS', rand(nStates, nStates));

            % Compute the training error
            vTargetsHMM_ = hmmviterbi(vTargetsDNN, mHMMTransitionProbs, mHMMEmissionProbs);
            
            % The input targets now are the outputs of this layer of HMM
            vTargetsDNN = vTargetsHMM_;
        end % end for
        
        % Accumulate total HMM outputs
        vTargetsHMM = [vTargetsHMM vTargetsHMM_];
        
        % Compute context error
        nTrainErr = size(find(vTargetsHMM_ ~= vTrainTargets(startIdx:endIdx)), 2);
        
        % Accumulate error
        nAccTrainErr = nAccTrainErr + nTrainErr;
        
        % Update the index of the next context
        startIdx = startIdx + sizeHMM(i);
        i = i + 1;

    end % end while

    nTrainErrFinal = size(find(vTargetsHMM ~= vTrainTargets), 2);

end % end function