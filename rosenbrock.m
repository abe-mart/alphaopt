close all; clear; clc;

% Setup optimization options
options = optimoptions(@fminunc,'Display','iter','Algorithm','quasi-newton');

% Define guess values
xy_guess = [0,0];

% Call optimization algorithm
[xy_opt,fval] = fminunc(@rosenbrock_func,xy_guess,options)

% Objective function
function f = rosenbrock_func(in)
    % Unpack inputs
    x = in(1);
    y = in(2);
    
    % The Rosenbrock function in 2D
    f = 100*(y - x^2)^2 + (1 - x)^2;

end