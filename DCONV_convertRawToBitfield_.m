% Function:
% Converts input Raw batch matrix into bitfield batch matrix
% Inputs:
% mRawBatchData: Matrix (nBatchSize X nMaxNumFeatures) with
% nNumFeatures locations correspond to "1" in the final bit-field
% nBitfieldLength: Length of the final bit-field
% Output:
% mBitFieldBatchData: Matrix (nBatchSize X nBitfieldLength X nNumBatches) final bit-field
function [mBitFieldBatchData] = DCONV_convertRawToBitfield(mRawBatchData, nBitfieldLength)

    % Initialize the return data
    mBitFieldBatchData = zeros(size(mRawBatchData, 1), nBitfieldLength);


    % Loop on examples rows per batch
    for i = 1 : size(mRawBatchData, 1)    
        % Set 1 locations for current example. The loop is only until
        % the non-zeros positions
        for m = 1 : size(find(mRawBatchData(i,:)~=0), 2)
            mBitFieldBatchData(i,mRawBatchData(i,m)) = 1;
        end
    end

end