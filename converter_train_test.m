fid = fopen('..\Preprocessing\features_gadwal_binary.txt');%
%fid = fopen('features_binary_final_tmp.txt');

train_features=[]; 
train_targets=[]; 

test_features=[]; 
test_targets=[]; 

target_feature = 1;
train_factor = 6;
%load arch_500_500_2000_1000_features_10_targets_kitchen_l train_factor;
mail_ctr = 0;
s = fgets(fid);

while(s > 0)
	D = str2num(s);
	if (mod(mail_ctr, train_factor) == 0)	
		if (target_feature == 1)
			test_targets = [test_targets; D];
		else
			test_features = [test_features; D];
			mail_ctr = mail_ctr + 1;
		end;
	else
		if (target_feature == 1)
			train_targets = [train_targets; D];
		else
			train_features = [train_features; D];
			mail_ctr = mail_ctr + 1;
		end;
	end;
	target_feature = ~target_feature;
	s = fgets(fid);
end;
fclose(fid);
save train_test_features_targets.mat test_features test_targets train_features train_targets;