function [top_cut,bottom_cut, th_new] = getLimits(x)
% GETLIMITS returns indecies used to trim vector X.
%   OUTPUTS:
%       TOP_CUT    =Index of the last "seen" Nan from the top of the well.
%       
%       BOTTOM_CUT =Index of the last "seen" row of zeros from the bottom
%       of the well.
%   	
%       TH_NEW     = top_cut - bottom_cut
% Written by: Obai Shaikh

th = length(x);
top_flag = 0;

for i = 1 : th
    if sum(isnan(x(i,:))) == 0 && top_flag == 0
        top_cut = i;
        top_flag = 1;
    elseif sum(x(i,:)) == 0 && top_flag == 1
        bottom_cut = i;
        break
    end 
    
end

th_new = bottom_cut - top_cut;
end

