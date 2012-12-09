function [error_train, error_val, arb] = ...
    learningCurve(X, y, Xval, yval, lambda, hidden_layer_size, hidden_layer2_size, hidden_layer3_size, input_layer_size, num_labels)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).


% Number of training examples
m = size(X, 1);
arb = 6;
% You need to return these values correctly
error_train = zeros(arb, 1);
error_val   = zeros(arb, 1);



v = floor(linspace(15,m, arb));
i = 1;
for j = v
	rng default;
	[theta1, theta2, theta3, theta4] = trainNN(X(1:j,:),y(1:j), lambda, hidden_layer_size, ...
													hidden_layer2_size, hidden_layer3_size, input_layer_size, num_labels);
	nn_params = [theta1(:) ; theta2(:); theta3(:); theta4(:)];
	error_train(i) = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
													hidden_layer2_size, hidden_layer3_size,	num_labels, X(1:j,:),y(1:j), 0);
	error_val(i) = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
													hidden_layer2_size, hidden_layer3_size,	num_labels, Xval, yval, 0);
	i = i + 1;
end;


end
