function p = predict(Theta1, Theta2, Theta3, Theta4, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta4, 1);
p = zeros(size(X, 1), 1);


h1 = sigmoid([ones(m, 1) X] * Theta1');
h2 = sigmoid([ones(m, 1) h1] * Theta2') ;
h3 = sigmoid([ones(m, 1) h2] * Theta3') ;
h4 = ([ones(m,1), h3] * Theta4');
class_normalizer = log_sum_exp_over_rows(h4');
tiri = repmat(class_normalizer, [size(h4, 2), 1]); 
log_class_prob = h4 - tiri';
class_prob = exp(log_class_prob); 


[dummy, p] = max(class_prob, [], 2);

% =========================================================================


end
