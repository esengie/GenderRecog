function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, hidden_layer2_size, hidden_layer3_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad is an "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
t1 = hidden_layer_size * (input_layer_size + 1);



Theta1 = reshape(nn_params(1:t1), ...
                 hidden_layer_size, (input_layer_size + 1));
t2 = t1 + hidden_layer2_size * (hidden_layer_size + 1);
Theta2 = reshape(nn_params((t1 + 1): t2), ...
                 hidden_layer2_size, (hidden_layer_size + 1));

t3 = t2 + hidden_layer3_size * (hidden_layer2_size + 1);			 
Theta3 = reshape(nn_params((t2 + 1): t3), ...
                 hidden_layer3_size, (hidden_layer2_size + 1));
	 
	 
Theta4 = reshape(nn_params((t3 + 1): end), ...
                 num_labels, (hidden_layer3_size + 1));

% Setup some useful variables
m = size(X, 1);
         
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
Theta3_grad = zeros(size(Theta3));
Theta4_grad = zeros(size(Theta4));


% dropout implementation: use at your own risk
% note: will not work with current implementation, probably needs 
% simple stochastic gradient descent with momentum not a complicated Line search

%dropout1 = abs(rand(size(Theta1,1), 1) - 0.2);
%dropout2 = abs(rand(size(Theta2,1), 1) - 0.5);
%dropout3 = abs(rand(size(Theta3,1), 1) - 0.5);
%dropout4 = abs(rand(size(Theta4,1), 1) - 0.2);
%dropout1 = repmat(dropout1, 1,size(Theta1,2));
%dropout2 = repmat(dropout2, 1,size(Theta2,2));
%dropout3 = repmat(dropout3, 1,size(Theta3,2));
%dropout4 = repmat(dropout4, 1,size(Theta4,2));
%Theta1 = Theta1 .* dropout1;
%Theta2 = Theta2 .* dropout2;
%Theta3 = Theta3 .* dropout3;
%Theta4 = Theta4 .* dropout4;

y = [y, zeros(m, num_labels-1)];
for i=1:m
	y(i,y(i,1)) = 1;
end;
y(:,1) = y(:,1) == 1;

h1 = sigmoid([ones(m,1), X] * Theta1');
h2 = sigmoid([ones(m,1), h1] * Theta2');
h3 = sigmoid([ones(m,1), h2] * Theta3');
h4 = ([ones(m,1), h3] * Theta4');

class_normalizer = log_sum_exp_over_rows(h4');
tiri = repmat(class_normalizer, [size(h4, 2), 1]); 
log_class_prob = h4 - tiri';
class_prob = exp(log_class_prob); 
%y  = 1 + mod(1:m, num_labels)';

J = -mean(sum(log_class_prob .* y, 2));

reg = lambda/(2*m) * (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)) + sum(sum(Theta3(:,2:end).^2)) + sum(sum(Theta4(:,2:end).^2)));
J = J + reg;


Delta5 = (class_prob - y);
Delta4 = Delta5 * Theta4 .* [ones(m,1), sigmoidGradient([ones(m,1), h2] * Theta3')];


Delta3 = Delta4(:,2:end) * Theta3 .* [ones(m,1), sigmoidGradient([ones(m,1), h1] * Theta2')];
Delta2 = Delta3(:,2:end) * Theta2 .* [ones(m,1), sigmoidGradient([ones(m,1), X] * Theta1')];

Theta1_grad = Theta1_grad + Delta2(:,2:end)' * [ones(m,1), X];
Theta2_grad = Theta2_grad + Delta3(:,2:end)' * [ones(m,1), h1];
Theta3_grad = Theta3_grad + Delta4(:,2:end)' * [ones(m,1), h2];
Theta4_grad = Theta4_grad + Delta5' * [ones(m,1), h3];


Theta1_grad = Theta1_grad / m;
Theta2_grad = Theta2_grad / m;
Theta3_grad = Theta3_grad / m;
Theta4_grad = Theta4_grad / m;

Theta1(:,1) = zeros(size (Theta1(:,1))); 
Theta2(:,1) = zeros(size (Theta2(:,1)));
Theta3(:,1) = zeros(size (Theta3(:,1)));
Theta4(:,1) = zeros(size (Theta4(:,1)));
Theta1_grad = Theta1_grad + Theta1 * lambda / m;
Theta2_grad = Theta2_grad + Theta2 * lambda / m;
Theta3_grad = Theta3_grad + Theta3 * lambda / m;
Theta4_grad = Theta4_grad + Theta4 * lambda / m;

%Theta1_grad = Theta1_grad .* dropout1;
%Theta2_grad = Theta2_grad .* dropout2;
%Theta3_grad = Theta3_grad .* dropout3;
%Theta4_grad = Theta4_grad .* dropout4;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:); Theta3_grad(:); Theta4_grad(:)];


end
