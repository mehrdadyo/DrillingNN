function [superSet] = logAug(X,window,stride, n)
%LOGAUG Well log Augmentation
%   Extracts subsets of shape (window, nLogs) from the given 3d matrix of
%   logs X.
%   
%   INPUTS:
%       X      = of shape (A, B, C) where A is the number of steps (log
%       depth axis), B is the number of curves, C is the number of wells.
% 
%       WINDOW = height of the scanning window.
% 
%       STRIDE = number of points to shift the window down the log
% 
%       N = Integer, specifies the length of the output vector SUPERSET.
%        
%   OUTPUT:
%       SUPERSET = Matrix containing subsets of all logs 
%       of shape ( window * No. of iterations, nLogs)
% Written by: Obai Shaikh
if nargin < 4
    n = 1e4;
end

dim = size(X,3);
nLogs = size(X,2);

superSet = zeros(window * n , nLogs);
iter = zeros(dim, 1);

for i = 1:dim
    
    log = X(:,:,i);
    
    % Trim zeros and nans:
    [top_cut,bottom_cut, ~] = getLimits(log);
    log = log(top_cut:bottom_cut-1,:);
    lo_cut = lonanFix(log);
    log = log(1:lo_cut,:);
    
    % remove rows with interwell nans:
    log(any(isnan(log), 2), :) = [];
    
    % Standardize:
    mu = mean(log,1);
    sigma = std(log,1) + .0001;
    log = (log - mu +.0001)./sigma;
    
    % Compute # of iterations for each log:
    th = length(log);
    iter(i) = floor((th - window)/stride+1);
    
    % Allocation:
    set = zeros(window *  iter(i), nLogs);
    
    % Extract subsets:
    for j = 1:1: iter(i)
        count = j-1;                        % start from 0
        vert_start = count * stride + 1;    % find chunk's dimensions
        vert_end = vert_start + window - 1;
        chunk = log(vert_start:vert_end,:); % get chunk
        
        from = count * window + 1;
        set(from:from + window-1,:) = chunk;% assign chunks
    end
    % Store subsets        
    if i == 1                        
        superSet(1: window * iter(i),:) = set;
        start = window * iter(i) +1;
    else
        superSet(start : start+ window*iter(i)-1 , :) = set;
        start = start+ window * iter(i);
    end
end

% Trim trailing zeros:
flag = sum(superSet,2);
superSet(flag == 0,:) = [];

end

