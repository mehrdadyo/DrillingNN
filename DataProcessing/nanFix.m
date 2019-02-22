function [x] = nanFix(x)
%NANFIX replaces singluar values of Nan withing a log (X) by avering the two
%surrounding values to that Nan and returns the corrected log.
% Written by: Obai Shaikh
len = length(x);

for i = len-1 :-1: 2
    if ~isnan(x(i+1)) && isnan(x(i)) && ~isnan(x(i-1))          % if nan, continue
       x(i) = (x(i+1) + x(i-1))/2;
    end
    
end

end

