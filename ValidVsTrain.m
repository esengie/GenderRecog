function [error_train, error_val, arb] = ...
    ValidVsTrain(X, y, Xval, yval, lambda, hidden_layer_size, hidden_layer2_size, hidden_layer3_size, input_layer_size, num_labels)
%VALIDVSTRAIN Generates the train and cross validation set errors needed 
%to plot a ValidVsTrain curve
%   [error_train, error_val] = ...
%       VALIDVSTRAIN(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i iteraions (and similarly for error_val(i)).
%

% Number of max iters
m = 150;
arb = 5;

error_train = zeros(arb, 1);
error_val   = zeros(arb, 1);

v = floor(linspace(15,m, arb));
i = 1;
for j = v
	rng default;
	opts = optimset('MaxIter', j);
	[theta1, theta2, theta3, theta4] = trainNN(X,y, lambda, hidden_layer_size, ...
													hidden_layer2_size, hidden_layer3_size, input_layer_size, num_labels, opts);
	nn_params = [theta1(:) ; theta2(:); theta3(:); theta4(:)];
	error_train(i) = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
													hidden_layer2_size, hidden_layer3_size,	num_labels, X,y, 0);
	error_val(i) = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
													hidden_layer2_size, hidden_layer3_size,	num_labels, Xval, yval, 0);
	i = i + 1;
end;

end
