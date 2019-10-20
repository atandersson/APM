%% Generate data

% parameters for response function
a    = 1/2;
b    = [2,2];
c    = [2,3];
N_TR = 1000;  % # of training examples
N_TE = 10^4;  % # of test examples
d    = 2;     % dimension of x

% generate training data
[x_TR,y_TR] = generateData(a,b,c,N_TR,d);
D_TR        = [y_TR,x_TR];

% generate test data
[x_TE,y_TE] = generateData(a,b,c,N_TE,d);
D_TE        = [y_TE,x_TE];

%% Batch-mode active learning

B         = 96;     % batch size
P         = 20;     % # of prediction models
t_max     = 10;     % # of batches
dataIndex = 1:N_TR; % store data indices

% 1. Set k
k = B;

% 2. Create a training set S
newIndex            = randperm(numel(dataIndex),B);
S                   = dataIndex(newIndex)';
dataIndex(newIndex) = [];

% 3. Define t
t = 1;

x1 = 0:0.5:6;

gaussianPeak = zeros(numel(x1));
for i = 1:numel(x1)
    for j = 1:numel(x1)
        gaussianPeak(i,j) = respFunc(a,b,c,[x1(i),x1(j)]');
    end
end

figure(1)
subplot(1,2,1)
surf(x1,x1,gaussianPeak')
xlabel('x1')
ylabel('x2')
title('Original Gaussian (3D)')
colorbar
subplot(1,2,2)
surf(x1,x1,gaussianPeak')
xlabel('x1')
ylabel('x2')
title('Original Gaussian (2D)')
view(2)
colorbar

figure(2)
subplot(2,5,1)
scatter3(x_TR(S,1),x_TR(S,2),y_TR(S),'*')
title(sprintf('S at t = %d',t))

Y    = zeros(numel(x1),numel(x1),t_max-1);
RMSE = zeros(1,t_max-1);

% 4. The while-loop
while t < t_max
    
    %----------(a)----------
    sets = randi(k,k,P);
    S_stars = S(sets);
    
    %----------(b)----------
    
    y_hat = zeros(size(y_TE,1));
    for p = 1:P
        S_star = S_stars(:,p);
        eval(sprintf("net%d = createNetworkModel(x_TR(S_star,:),y_TR(S_star));",p))
        for i = 1:numel(x1)
            for j = 1:numel(x1)
                eval(sprintf("Y(i,j,t) = Y(i,j,t) + net%d([x1(i);x1(j)]);",p))
            end
        end
        eval(sprintf("y_hat = y_hat + net%d(x_TE')';",p))
    end
    y_hat = y_hat/P;
    RMSE  = sqrt(sum((y_hat-y_TE).^2)/numel(y_TE))
    
    %----------(c)----------
    k = B*(t+1);
    
    %----------(d) to (e)----------
    newIndex            = randperm(numel(dataIndex),B);
    S                   = [S;dataIndex(newIndex)'];
    dataIndex(newIndex) = [];
    
    %----------(i)----------
    t = t+1;
    
    figure(2)
    subplot(2,5,t)
    scatter3(x_TR(S,1),x_TR(S,2),y_TR(S),'*')
    title(sprintf('S at t = %d',t))
end

Y = Y/20;

figure(3)
for i = 1:(t_max-1)
    subplot(3,3,i)
    surf(x1,x1,Y(:,:,i)')
    xlabel('x1')
    ylabel('x2')
    title(sprintf('Average surface estimation (t=%d)',i))
    colorbar
end

figure(4)
for i = 1:(t_max-1)
    subplot(3,3,i)
    surf(x1,x1,Y(:,:,i)')
    xlabel('x1')
    ylabel('x2')
    title(sprintf('Average surface estimation (t=%d)',i))
    colorbar
    view(2)
end

labels = 2:t_max;

data = mean(RMSE)';

figure(5)
bar(labels,data)                
title('RMSE (1\sigma)')

figure(6)
bar(labels,data)                
title('RMSE (1.96\sigma)')