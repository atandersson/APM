% parameters for y
a    = 1/2;
b    = [2,2];
c    = [2,3];
N_TR = 200;   % # of training examples
N_TE = 10^4;  % # of test examples
d    = 2;     % dimension of x

% generate and store data
[x_TR,y_TR] = generateData(a,b,c,N_TR,d);
[x_TE,y_TE] = generateData(a,b,c,N_TE,d);