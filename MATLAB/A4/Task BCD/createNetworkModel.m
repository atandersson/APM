function trained_net = createNetworkModel(X,y)

    % DEFINE THE NETWORK ARCHITECTURE
    numInputs          = 1;     % NOTE: THIS DOES NOT MEAN THAT THERE IS ONLY ONE INPUT 
                                %       DIMENSION TO THE NET! THE DIMENSIONS IS ASSIGNED
                                %       BELOW AFTER NET INITIALIZATION
    numLayers          = 2;
    biasConnect        = [1;
                          1]; % - numLayers-by-1 Boolean vector, zeros.
    inputConnect       = [1;
                          0];   % - numLayers-by-numInputs Boolean matrix, zeros.
    layerConnect       = [0 0;
                          1 0]; % - numLayers-by-numLayers Boolean matrix, zeros.
    outputConnect      = [0 1]; % - 1-by-numLayers Boolean vector, zeros.
    net                = network(numInputs,numLayers,biasConnect,inputConnect,layerConnect,outputConnect);
    no_of_hidden_nodes = 100;     %THIS IS THE NUMBER OF HIDDEN NODES IN THE NET TO BE TRAINED

    net.layers{1}.dimensions = no_of_hidden_nodes;
    net.layers{2}.dimensions = 1;
    net.inputs{1}.size       = size(X,2);   % dimensions of input vectors
    
    % START WEIGHTS
    std_start_weights = 10^(-2);
    
    % W guess
    w1          = [-1  0]';
    w2          = [ 1  0]';
    w3          = [ 0 -1]';
    w4          = [ 0  1]';
    w5          = [-1 -1]';
    w6          = [ 1  1]';
    w7          = [ 1 -1]';
    w8          = [-1  1]';
    W_guess     = [w1,w2,w3,w4,w5,w6,w7,w8];
    W_remainder = (rand(2,no_of_hidden_nodes-size(W_guess,2)))*std_start_weights;
    W_guess     = [W_guess,W_remainder]';
    
    % a guess
    a_guess = std_start_weights*randn(1,no_of_hidden_nodes);
    
    % b guess
    a =  3.5;
    b = -2.5;
    c =  2.5;
    d = -1.5;
    e =  5.0;
    f = -4.5;
    g = -0.5;
    h =  1.5;
    
    b_guess     = [a; b; c; d; e; f; g; h];
    b_remainder = (rand(no_of_hidden_nodes-size(b_guess,1),1))*std_start_weights;
    b_guess     = [b_guess; b_remainder];

    % y guess
    yo_guess          = std_start_weights*randn;
    
    net.IW{1,1}       = W_guess;
    net.LW{2,1}       = a_guess;
    net.b{1}          = b_guess;
    net.b{2}          = yo_guess;

    % MAKE HIDDEN LAYER NON-LINEAR (DEFAULT IS 'purelin')
    net.layers{1}.transferFcn = 'tansig';

    % TRAIN THE NETWORK

    % Training function
    net.trainFcn = 'traingdx';  %Gradient descent with momentum and adaptive steps

    % Set up Division of Data for Training, Validation, Testing
    % NOTE: Here we do not use any built-in validation or test
    net.divideFcn              = 'dividerand';
    net.divideParam.trainRatio = 100/100;
    net.divideParam.valRatio   = 0/100;
    net.divideParam.testRatio  = 0/100;

    % Set training parameters
    initial_learning_rate      = 10^(-8);
    max_no_of_epochs           = 20000;
    performance_goal           = 10^(-20);
    minimum_gradient_threshold = 10^(-20); 

    net.trainParam.lr       = initial_learning_rate;
    net.trainParam.epochs   = max_no_of_epochs;
    net.trainParam.goal     = performance_goal;
    net.trainParam.min_grad = minimum_gradient_threshold;

    % Call train which performs the actual training
    trained_net = train(net,X',y');
end