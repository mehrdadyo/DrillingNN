function [superSet, swin] = logAug(X,window,stride, n, Hdrs)
%LOGAUG Well log Augmentation
%   Extracts subsets of shape (window, nLogs) from the given 3d matrix of
%   logs X.
%   Rows containing at lease one Nan from the start or end of the well will
%   be removed. Nans within the well will be interpolated linearly from
%   surrounding values.
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
    
    % Trim zeros and nans from top and bottom of well:
    [top_cut,bottom_cut, th, th_new] = getLimits(log);
    log = log(top_cut:bottom_cut-1,:);
    lo_cut = lonanFix(log);
    log = log(1:lo_cut,:);
    
    disp(['No. of points:  ',num2str(th_new), '      New Thickness:  ',...
        num2str(th_new.*0.15)])

%     disp(['Top trim: ', num2str(top_cut), '   Bottom trim: ', ...
%         num2str(bottom_cut)])
%     disp(['Low nan cut: ', num2str(lo_cut)])
    
    %linear interp. inter-well nans in each column
    for j=2:nLogs     
        x = log(:,j);
        nanx = isnan(x);
        t    = 1:numel(x);
        x(nanx) = interp1(t(~nanx), x(~nanx), t(nanx));
        log(:,j) = x;
    end

    % Apply Low-pass filter:
%     p = 100;
%     lo_pass = ones(1, p)/p;
%     log = filter(lo_pass, 1, log);
    [log, swin] = smoothdata(log,1, 'sgolay',32);
    
    if mod(i,2)==0
        figure;lplot(log,Hdrs);title(['Well: ', num2str(i)]);
    end
    % Standardize:
%     mu = mean(log,1);
%     sigma = std(log,1) + .0001;
%     log = (log - mu +.0001)./sigma;
%     mu_rho(i) = mu(end-1);
%     std_rho(i) = sigma(end-1);
    
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

