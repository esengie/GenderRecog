clear ; close all; clc

%% Setup the parameters 
input_layer_size  = 900;  
hidden_layer_size = 700; 
hidden_layer2_size = 450;
hidden_layer3_size = 200;
num_labels = 2;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)

%% =========== Part 1: Loading and Visualizing Data =============

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

load('X.mat');

%X = gpuArray(X);
%y = gpuArray(y);

%normer = max(abs(X));
%X = bsxfun(@rdivide, X, normer);

temp = find(y == 2);
numel(temp);
numb = ceil((max(temp) - min(temp))*1.7);
yiu = (randperm(numel(y) - numel(temp)))';
glue = [yiu(1:numb); temp];
Xval = X(yiu(numb+1:end),:);
yval = y(yiu(numb+1:end));
X = X(glue,:);
y = y(glue);

m = size(X, 1);
glue = randperm(m);
X = X(glue,:);
y = y(glue,:);

validation = ceil(0.3 * m);
Xval = [Xval; X(m - validation:end,:)];
yval = [yval; y(m - validation:end,:)];
X = X(1 : m - validation - 1,:);
y = y(1 : m - validation - 1,:);



% Randomly select 100 data points to display
%sel = randperm(size(X, 1));
%sel = sel(1:100);

%ui = giveBack(X(sel,:));
%displayData(ui);

%%%%%%%%%%%%Gradient Checker%%%%%%%%%%%%%%%%%%%%%%%%
%checkNNGradients;
%%%%%%%%%%%%%%%%

%lambda = 3;
%checkNNGradients(lambda);



%% =================== Training NN ===================

fprintf('\nTraining Neural Network... \n')


lambda = 3;
[Theta1, Theta2, Theta3, Theta4] = trainNN(X, y, lambda, hidden_layer_size, hidden_layer2_size, hidden_layer3_size, input_layer_size, num_labels);
save('thetas.mat', 'Theta1', 'Theta2', 'Theta3', 'Theta4');

%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! validVstrain curve

%lambda = 3;
%figure(3);
%[error_train, error_val, arb] = ...
%   ValidVsTrain(X, y, Xval, yval, lambda, hidden_layer_size, hidden_layer2_size, hidden_layer2_size, input_layer_size, num_labels);
%plot(1:size(error_train), error_train, 1:size(error_val), error_val);

%title(sprintf('Polynomial Regression ValidVsTrain Curve (lambda = %f)', lambda));
%xlabel('Number of iterations')
%ylabel('Error')
%axis([0 arb+1 0 2])
%legend('Train', 'Cross Validation')
%fprintf('Polynomial Regression (lambda = %f)\n\n', lambda);
%fprintf('# Iterations\tTrain Error\tCross Validation Error\n');
%j = 1;
%for i = floor(linspace(1,m, arb))
%    fprintf('  \t%d\t\t%f\t%f\n', i, error_train(j), error_val(j));
%	j = j + 1;
%end

%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! learn curve
%lambda = 3;
%figure(3);
%[error_train, error_val, arb] = ...
%   learningCurve(X, y, Xval, yval, lambda, hidden_layer_size, hidden_layer2_size, hidden_layer2_size, input_layer_size, num_labels);
%plot(1:size(error_train), error_train, 1:size(error_val), error_val);

%title(sprintf('Polynomial Regression Learning Curve (lambda = %f)', lambda));
%xlabel('Number of training examples')
%ylabel('Error')
%axis([0 arb+1 0 2])
%legend('Train', 'Cross Validation')
%fprintf('Polynomial Regression (lambda = %f)\n\n', lambda);
%fprintf('# Training Examples\tTrain Error\tCross Validation Error\n');
%j = 1;
%for i = floor(linspace(1,m, arb))
%    fprintf('  \t%d\t\t%f\t%f\n', i, error_train(j), error_val(j));
%	j = j + 1;
%end


%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! valid curve
%[lambda_vec, error_train, error_val] = ...
%   validationCurve(X, y, Xval, yval, hidden_layer_size, hidden_layer2_size, hidden_layer3_size, input_layer_size, num_labels);

%figure(4);
%plot(lambda_vec, error_train, lambda_vec, error_val);
%legend('Train', 'Cross Validation');
%xlabel('lambda');
%ylabel('Error');

%fprintf('lambda\t\tTrain Error\tValidation Error\n');
%for i = 1:length(lambda_vec)
%	fprintf(' %f\t%f\t%f\n', ...
%          lambda_vec(i), error_train(i), error_val(i));
%end



pred = predict(Theta1, Theta2, Theta3, Theta4, Xval);

fprintf('\nValidation Set Accuracy: %f\n', mean(double(pred == yval)) * 100);

pred = predict(Theta1, Theta2, Theta3, Theta4, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
