mClassFeatures = [];
for u = 1 : size(DPREP_strData.mTrainFeatures,1) 
    if(vTrainTargets(u) == 8) 
        mClassFeatures = [mClassFeatures; DPREP_strData.mTrainFeatures(u,:)]; 
    end
end

mClassFeatures_8 = mClassFeatures;