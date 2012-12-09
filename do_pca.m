clear ; close all; clc

fprintf('\nLoading face dataset.\n\n');

%  Load Face dataset
load ('X_Y.mat');

%  Display the first 100 faces in the dataset
displayData(X(1:25, :));

%  Run PCA and visualize the eigenvectors which are in this case eigenfaces
%  We display the first 36 eigenfaces.
%
fprintf(['\nRunning PCA on face dataset.\n' ...
         '(this mght take a minute or two ...)\n\n']);

%  Before running PCA, it is important to first normalize X by subtracting 
%  the mean value from each feature
[X_norm, mu, sigma] = featureNormalize(X);

% I had more man faces sorry this is a stupid workaround
%__________________________________________
temp = find(y == 2);
numel(temp);
numb = max(temp) - min(temp);
glue = [(randperm(numb))'; temp];
%__________________________________________

%  Run PCA
[U, S] = pca(X_norm(glue,:));

pause;
%  Visualize the top 36 eigenvectors found
displayData(U(:, 1:36)');

fprintf('Program paused. Press enter to continue.\n');
pause;


%% Dimension Reduction for Faces
%  Project images to the eigen space using the top k eigenvectors 
%  If you are applying a machine learning algorithm 
fprintf('\nDimension reduction for face dataset.\n\n');

K = 900;
Z = projectData(X_norm, U, K);

fprintf('The projected data Z has a size of: ')
fprintf('%d ', size(Z));

fprintf('\n\nProgram paused. Press enter to continue.\n');
pause;

%% Visualization of Faces after PCA Dimension Reduction
%  Project images to the eigen space using the top K eigen vectors and 
%  visualize only using those K dimensions
%  Compare to the original input, which is also displayed

fprintf('\nVisualizing the projected (reduced dimension) faces.\n\n');

X_rec  = recoverData(Z, U, K);

% Display normalized data
subplot(1, 2, 1);
displayData(X_norm(1:25,:));
title('Original faces');
axis square;

% Display reconstructed data from only k eigenfaces
subplot(1, 2, 2);
displayData(X_rec(1:25,:));
title('Recovered faces');
axis square;

fprintf('Program paused. Press enter to continue.\n');
pause;

X = Z;
save ('X_norm.mat', 'mu', 'sigma', 'U', 'K');
save ('X.mat', 'X', 'y');