function ret = giveBack(X, U, K, sigma, mu)

	X = recoverData(X, U, K);
	X = bsxfun(@times, X, sigma);
	X = bsxfun(@plus, X, mu);
	ret = X;
end