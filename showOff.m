clear; close all; clc;

load ('X.mat');
load ('thetas.mat');
load ('X_norm.mat');
pred = predict(Theta1, Theta2, Theta3, Theta4, X);

ind1 = find(pred ~= 1);
ind2 = find(pred ~= y);
ind = intersect(ind1, ind2);

X_rec = giveBack(X, U, K, sigma, mu);
i = 1;
while i + 30 < numel(ind)
	teo = ind(i:i+25);
	displayData(X_rec(teo,:));
	pause;
	i = i + 30;
end;