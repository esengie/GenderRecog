function W = randInitializeWeights(L_in, L_out)
%RANDINITIALIZEWEIGHTS Randomly initialize the weights of a layer with L_in
%incoming connections and L_out outgoing connections
%   W = RANDINITIALIZEWEIGHTS(L_in, L_out) randomly initializes the weights 
%   of a layer with L_in incoming connections and L_out outgoing 
%   connections. 
%

W = zeros(L_out, 1 + L_in);

% Initialize W randomly so that we break the symmetry while
% training the neural network.
%

epsiloninit = sqrt(6) / sqrt(L_in + L_out);
W = rand(L_out, 1 + L_in)* 2 * epsiloninit - epsiloninit;

% =========================================================================

end
