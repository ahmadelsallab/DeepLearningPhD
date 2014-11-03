load input_data
max_num_ones = 0;
for i = 1 : size(mTrainFeatures, 1)
    num_ones(i) = size(find(mTrainFeatures(i,:)~=0),2);
end

fprintf(1, 'Max = %d\n', max(num_ones));
fprintf(1, 'Min = %d\n', min(num_ones));
fprintf(1, 'Mean = %d\n', mean(num_ones));