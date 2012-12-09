function [lambda_vec, error_train, error_val] = ...
    validationCurve(X, y, Xval, yval, hidden_layer_size, hidden_layer2_size, hidden_layer3_size, input_layer_size, num_labels)
%VALIDATIONCURVE Generate the train and validation errors needed to
%plot a validation curve that we can use to select lambda
%   [lambda_vec, error_train, error_val] = ...
%       VALIDATIONCURVE(X, y, Xval, yval) returns the train
%       and validation errors (in error_train, error_val)
%       for different values of lambda. You are given the training set (X,
%       y) and validation set (Xval, yval).
%

% Selected values of lambda (you should not change this)
lambda_vec = [0.001 0.003 0.01 0.03 0.1 0.3 1 3 10 30]';

error_train = zeros(length(lambda_vec), 1);
error_val = zeros(length(lambda_vec), 1);


for i = 1:length(lambda_vec)
    lambda = lambda_vec(i);
	rng default;
	[theta1, theta2, theta3, theta4] = trainNN(X,y, lambda, hidden_layer_size, hidden_layer2_size, hidden_layer3_size, input_layer_size, num_labels);
	nn_params = [theta1(:) ; theta2(:); theta3(:); theta4(:)];
	error_train(i) = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, hidden_layer2_size, hidden_layer3_size, num_labels, X, y, 0);
	error_val(i) = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, hidden_layer2_size, hidden_layer3_size, num_labels, Xval, yval, 0);

end
