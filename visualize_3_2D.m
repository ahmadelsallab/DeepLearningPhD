clear, clc;

%2D visualize%
N_2 = 4; % number of points to plot
N_dim = 2;% dimension
code = zeros(N_2,N_dim);
code(:,1)=[1 2 2 3];
code(:,2)=[1 1 2 4];
axis([0 10 0 10]);
hold on;
plot(code(:,1), code(:,2),'+');
hold off;
figure;

%3D visualize%
N_3 = 4; % number of points to plot
N_dim = 3; % dimension

code = zeros(N_3,N_dim);
code(:,1)=[1 2 2 2];
code(:,2)=[1 2 2 3];
code(:,3)=[1 2 1 2];
axis([0 10 0 10 0 10]);
hold on;
plot3(code(:,1), code(:,2), code(:,3),'+');
hold off;