% Function:
% Trains the TFIDF classifier
% Inputs:
% mTrainFeatures: Input data. Matrix (nxm), where n is the number of examples and m is the features vector length
% mTrainTargets: Associated targets. Vector (nxl), where n is the number of examples and l is the number of target classes
% Output:
% TFIDF_clsParams: The TFIDF weights
function [TFIDF_clsParams] = CLS_trainTFIDF(mTrainFeatures, mTrainTargets)

    % First, build the term frequency matrix TF
    mTermFrequency = zeros(size(mTrainTargets ,2), size(mTrainFeatures, 2));
    
    % Get the train target vector
    [I vTrainTargets]=max(mTrainTargets, [], 2);
        
    for i = 1 : size(mTrainFeatures, 1)
        mTermFrequency(vTrainTargets(i), :) = mTermFrequency(vTrainTargets(i), :) + mTrainFeatures(i, :);
    end % end for
    
    % Calculate the logarithmic TF matrix
    %mLogTF = (1 + log10(mTermFrequency + 1));
    mLogTF = (mTermFrequency + 1);
    
    % Get the document frequency vector
    vDocFrequency = sum(mTrainFeatures, 1) + 1;
    
    % Get inverse document freqncy IDF
    %mIDF = log10(1./vDocFrequency);
    mIDF = (1./vDocFrequency);
    
    % Calculate the Wd,t = (1 + log10(tf))xlog10(N=8/df)
    %TFIDF_clsParams.mWeightsTFIDF = mLogTF * mIDF';
    TFIDF_clsParams.mWeightsTFIDF = zeros(size(mLogTF, 1), size(mIDF, 2));
    for j = 1 : size(mLogTF, 1)
        TFIDF_clsParams.mWeightsTFIDF(j,:) = mLogTF(j,:).*mIDF;
    end
    
end % end function