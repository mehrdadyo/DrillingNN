function [bottom_cut] = lonanFix(x)
%LONANFIX return the index of the last seen Nan from the bottom of the 
%   given vector X 

% Written by: Obai Shaikh

th = length(x);

for i = th:-1:1

    if sum(isnan(x(i,:))) == 0 
    bottom_cut = i;
    break;
    end
    
end

end

