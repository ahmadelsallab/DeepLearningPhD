clear, clc;
cd 'D:\Work\Research\PhD\Implementation\Diactrization\Results\Configuration_0.79';
load input_data mTestTargets mTestFeatures nBitfieldLength vChunkLength vOffset;

load final_net;
NM_strNetParams_Global = NM_strNetParams;

% fat7a, kasra, damma
cd 'D:\Work\Research\PhD\Implementation\Diactrization\Results\Configuration_0.78';
load final_net;
NM_strNetParams_Sub = NM_strNetParams;


% fat7ten, kasreten, dammeten
cd 'D:\Work\Research\PhD\Implementation\Diactrization\Results\Configuration_0.82';
load final_net;
NM_strNetParams_Sub_2 = NM_strNetParams;

% kasra, kasreten
cd 'D:\Work\Research\PhD\Implementation\Diactrization\Results\Configuration_0.84';
load final_net;
NM_strNetParams_Sub_3 = NM_strNetParams;

cd 'D:\Work\Research\PhD\Implementation\Generic Classifier\Generic_Classifier_0.13';

ctrCorrectlyClassified = 0;
ctrTotal = 0;

for i = 1 : size(mTestFeatures, 1)
    [I1 vTargetOut]=max(mTestTargets(i,:), [], 2); % J1 is the index where max. output is found in the desired target
    
    [nErr, vTestTargetsOutGlobal] = TST_computeClassificationErrDNN(mTestFeatures(i,:), mTestTargets(i,:), NM_strNetParams_Global, 0, 0, 0, 1, 1, 'ABSOLUTE_ERR_CALC', nBitfieldLength, vChunkLength, vOffset, 'Raw');

%%%%FAT7a, Damma,Kasra, Fat7ten,Dammeten,Kasreten-->Kasra, Kasreten%%%%
    %if(vTargetOut == 1 || vTargetOut == 2 || vTargetOut == 3)
    %if(vTargetOut == 4 || vTargetOut == 5 || vTargetOut == 6)
    if(vTestTargetsOutGlobal == 4 || vTestTargetsOutGlobal == 5 || vTestTargetsOutGlobal == 6)
        
        ctrTotal = ctrTotal + 1;
        
        % Fat7ten,Dammeten,Kasreten-->0.856
%         [nErr, vTestTargetsOutSub] = TST_computeClassificationErrDNN(mTestFeatures(i,:), mTestTargets(i,1:3), NM_strNetParams_Sub_2, 0, 0, 0, 1, 1, 'ABSOLUTE_ERR_CALC', nBitfieldLength, vChunkLength, vOffset, 'Raw');
% 
%         ctrCorrectlyClassified = ctrCorrectlyClassified + (vTargetOut==vTestTargetsOutSub);            

        % FAT7a, Damma,Kasra-->0.9024
        [nErr, vTestTargetsOutSub] = TST_computeClassificationErrDNN(mTestFeatures(i,:), mTestTargets(i,1:3), NM_strNetParams_Sub, 0, 0, 0, 1, 1, 'ABSOLUTE_ERR_CALC', nBitfieldLength, vChunkLength, vOffset, 'Raw');

        ctrCorrectlyClassified = ctrCorrectlyClassified + (vTargetOut==(vTestTargetsOutSub+3));  
        
        
%     elseif(vTestTargetsOutGlobal == 4 || vTestTargetsOutGlobal == 5 || vTestTargetsOutGlobal == 6)
%         
%         [nErr, vTestTargetsOutSub] = TST_computeClassificationErrDNN(mTestFeatures(i,:), mTestTargets(i,1:3), NM_strNetParams_Sub, 0, 0, 0, 1, 1, 'ABSOLUTE_ERR_CALC', nBitfieldLength, vChunkLength, vOffset, 'Raw');           
%         vTestTargetsOutSub = vTestTargetsOutSub + 3;
%             
%         % kasra is mosty confused with damma and fat7a
%         if(vTestTargetsOutGlobal == 6)
%             [nErr, vTestTargetsOutSub] = TST_computeClassificationErrDNN(mTestFeatures(i,:), mTestTargets(i,1:2), NM_strNetParams_Sub_3, 0, 0, 0, 1, 1, 'ABSOLUTE_ERR_CALC', nBitfieldLength, vChunkLength, vOffset, 'Raw');
%             if(vTestTargetsOutSub == 1)
%                 vTestTargetsOutSub = 3;
%             else
%                 vTestTargetsOutSub = 6;
%             end
%         end
%         
%         ctrCorrectlyClassified = ctrCorrectlyClassified + (vTargetOut==vTestTargetsOutSub);
%     else
%         ctrCorrectlyClassified = ctrCorrectlyClassified + (vTargetOut==vTestTargetsOutGlobal);
     end

    %ctrCorrectlyClassified = ctrCorrectlyClassified + (vTargetOut==vTestTargetsOutGlobal);
    
    fprintf(1, 'Accuracy = %d\n', ctrCorrectlyClassified / ctrTotal);
end
fprintf(1, 'Accuracy = %d\n', ctrCorrectlyClassified / ctrTotal);