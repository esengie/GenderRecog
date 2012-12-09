function [Theta1, Theta2, Theta3, Theta4] = trainNN(X, y, lambda, hidden_layer_size, hidden_layer2_size, hidden_layer3_size, input_layer_size, num_labels, opts)

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, hidden_layer2_size);
initial_Theta3 = randInitializeWeights(hidden_layer2_size, hidden_layer3_size);
initial_Theta4 = randInitializeWeights(hidden_layer3_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:) ; initial_Theta3(:) ; initial_Theta4(:)];



if nargin == 8
	options = optimset('MaxIter', 150);
else 
	options = opts;
end;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
								   hidden_layer2_size, hidden_layer3_size, ...
                                   num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);



% Obtain Theta.. back from nn_params
t1 = hidden_layer_size * (input_layer_size + 1);
Theta1 = reshape(nn_params(1:t1), ...
                 hidden_layer_size, (input_layer_size + 1));
t2 = t1 + hidden_layer2_size * (hidden_layer_size + 1);
Theta2 = reshape(nn_params((t1 + 1): t2), ...
                 hidden_layer2_size, (hidden_layer_size + 1));

t3 = t2 + hidden_layer3_size *(hidden_layer2_size + 1);			 
Theta3 = reshape(nn_params((t2 + 1): t3), ...
                 hidden_layer3_size, (hidden_layer2_size + 1));
	 
	 
Theta4 = reshape(nn_params((t3 + 1): end), ...
                 num_labels, (hidden_layer3_size + 1));

end