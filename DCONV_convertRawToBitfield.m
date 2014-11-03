% Function:
% Converts input Raw batch matrix into bitfield batch matrix
% Inputs:
% mRawBatchData: Matrix (nBatchSize X nMaxNumFeatures) with
% nNumFeatures locations correspond to "1" in the final bit-field
% nBitfieldLength: Length of the final bit-field
% vChunkLength: Vector storing the boundaries of each features chunk
% vOffset: The bitfield boundaries corresponding to each chunk
% Output:
% mBitFieldBatchData: Matrix (nBatchSize X nBitfieldLength X nNumBatches) final bit-field
function [mBitFieldBatchData] = DCONV_convertRawToBitfield(mRawBatchData, nBitfieldLength, vChunkLength, vOffset)

    % Initialize the return data
    mBitFieldBatchData = zeros(size(mRawBatchData, 1), nBitfieldLength);
  
    % Loop on examples rows per batch
    for i = 1 : size(mRawBatchData, 1)    
        % Set 1 locations for current example. The loop is only until
        % the non-zeros positions
        vNonZeros = find(mRawBatchData(i,:)~=0);
        for m = 1 : size(vNonZeros, 2)
            
            % Get the range of the ID from the boundaries of features chunks
%             k = 0;% vChunkLength(k) has the first entry always = 0, so any non-zero index must be > vChunkLength(0), so k at least will be incremented once
%             while(vNonZeros(m) > vChunkLength(k+1) && k < size(vChunkLength, 2))
%                k = k + 1;
%             end % while
            
            % Get the range of the ID from the boundaries of features chunks
            k = 0;% vChunkLength(k) has the first entry always = 0, so any non-zero index must be > vChunkLength(0), so k at least will be incremented once
            for (n = 1 : size(vChunkLength, 2))
                if(vNonZeros(m) > vChunkLength(n))
                    k = k + 1;
                else
                    break;
                end
            end
            
            % Get the poosition in bitfield
            nPositionInBitField = mRawBatchData(i, vNonZeros(m)) + vOffset(k);
            
            % Set the place in the bitfield to 1
            mBitFieldBatchData(i, nPositionInBitField) = 1;
            if(nPositionInBitField > nBitfieldLength)
                fprintf(1, 'error position in bitfield %d while bitfield length is %d\n', nPositionInBitField, nBitfieldLength);
            end
        end % for m = 1 : size(vNonZeros, 2)
    end % for i = 1 : size(mRawBatchData, 1) 
    
 %   load('e:\documents and settings\ASALLAB\Desktop\Flash\PhD_Flash\Implementation\Diactrization\Results\Configuration_0.76\tf_idf_trained');
%    mBitFieldBatchData = bsxfun(@times, mBitFieldBatchData, TFIDF_clsParams.vVarWeights);

end