function [x] = bitFix(x)

%BITFIX Similar to NANFIX. BitFix replaces singular Nan values withing the
%bitsize log by replacing that Nan value with the BitSize value below it.
%This function returns the corrected BitSize log X.

% Written by: Obai Shaikh

len = length(x);
% prev = x(len);

for i = len-1 :-1: 2
    if ~isnan(x(i+1)) && isnan(x(i)) && ~isnan(x(i-1))          % if nan, continue
       x(i) = x(i+1);
    end
        
    if isnan(x(i+1)) || isnan(x(i)) || isnan(x(i-1))          % if nan, continue
        continue
    else                    % if real No., check values
        up = x(i-1);
        current = x(i);
        down = x(i+1);
        
        if up > down && current ~= down && current ~= up
            x(i) = down;
            
        end
    end
    
end
end

