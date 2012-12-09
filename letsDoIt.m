%% Initialization
clear ; close all; clc

input_layer_size  = 6400;  % 80x80 Input Images of Digits


num_labels = 2;    
                         
% Load Training Data

cd ('testing/lets/');
D = dir('*.jpg');
X = zeros(numel(D), input_layer_size);

for i = 1:numel(D)
 temp = (imread(D(i).name));
  %if isrgb(temp)
	temp = rgb2gray(temp);
 % end;
  temp = imresize(temp,[80,80]);
  
  X(i,:) = (temp(:))';
end;

clear temp;

X = X / 256;
cd('../..');
load ('thetas.mat');
load ('X_norm.mat');

X_norm = bsxfun(@minus, X, mu);
X_norm = bsxfun(@rdivide, X_norm, sigma);

X = projectData(X_norm, U, K);
pred = predict(Theta1, Theta2, Theta3, Theta4, X);
X_rec = giveBack(X, U, K, sigma, mu);

for i = 1:size(X,1)
	displayData(X_rec(i,:));
	if pred(i) == 1
		title('male');
	else
		title('female');
	end;
	pause;
end;


