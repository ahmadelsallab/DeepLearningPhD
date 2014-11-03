% Convert

        %[mBitFieldBatchData] = DCONV_convertRawToBitfield(mRawBatchData, nBitfieldLength, vChunkLength, vOffset)

% Compute rr
err = 0;
for i = 1 : size(mData_1)
    err = err + sum(abs(mData_1(i,:) - mData_2(i,:)));
end
err