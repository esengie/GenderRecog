
%% Initialization
clear ; close all; clc

input_layer_size  = 6400;  % 80x80 Input Images of Digits


num_labels = 2;        

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

cd ('testing/1/');
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
y = ones (numel(D), 1);
clear temp;

cd ('../2/');
D2 = dir('*.jpg');
X2 = zeros(numel(D2), input_layer_size);
for i = 1:numel(D2)
  temp = imread(D2(i).name);
 % if isrgb(temp)
	temp = rgb2gray(temp);
 %end;
  temp = imresize(temp,[80,80]);
  X2(i,:) = (temp(:))';
end;
X = [X; X2];
X = X / 256;
y2 = ones(numel(D2), 1) * 2;
y = [y; y2];
clear X2, clear y2;
cd('../..');

save ('X_Y.mat', 'X', 'y');