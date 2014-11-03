    
    i = 1;
    x{i} = [1 2 3];
    i = 2;
    z = [1 2];
    x{i} = z;
    m = 1;
    savefile = ['pqfile' num2str(m) '.mat'];
    p = rand(1, 10);
    q = ones(10);
    save(savefile);
    
    clear;
    
    m=1;
    savefile = ['pqfile' num2str(m) '.mat'];
    
    load(savefile);