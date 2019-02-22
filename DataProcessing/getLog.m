function [log] = getLog(labels,data, headers)
%GETLOG returns desired logs specified in cell array LABELS extracted from
%the original log DATA with corresponding HEADERS
% Written by: Obai Shaikh

log = zeros(size(data,1), numel(labels));

for n = 1:numel(labels)
   d=strfind(headers,labels{n});
   ind = find(not(cellfun('isempty',d)));
   log(:,n) = data(:,ind(1));
end

end

