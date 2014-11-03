function [f, df] = f_test(x, y);

f = x;
df = y;
df = 3;
df = 2;
df = df - 1;
i = 3;
while(i>=1)
    fprintf(1,'Hello %d\n',i);
    i = i - 1;
end