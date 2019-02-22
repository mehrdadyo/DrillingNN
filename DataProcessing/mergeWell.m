function [t, labels,k] = mergeWell(mwdID,labels,compID, opt)
%MERGEWELL reads specific logs given by LABELS from the MWD logs in
% .las format given as fileIDs specified in the cell array MWDID. Then, 
% it concatenates such logs with their corresponding bitsize and gamma 
% ray logs from the composite logs given in the cell array COMPID.
% INPUTS:
%   MWDID: Cell array with .las fileIDs
%   LABELS: Headers of logs to extract from MWD logs
%   COMPID: Cell array of composite logs file IDs (must be same length as
%           MWDID.
%   OPT:    option of extraction method:
%           1           = use loadlas()
%           otherwise   = use read_las_file()
% Written by: Obai Shaikh

if nargin < 4       % setting default, if opt not entered
    opt =1;
end

t = zeros(3.01e4,numel(labels)+2, numel(mwdID));
k= zeros(numel(mwdID),1);
for id = 1:numel(mwdID)
    
    if opt == 1
        [~,data,header,~]=loadlas(mwdID{id});
    else
        wlog=read_las_file(mwdID{id});
        data = wlog.curves;
        header = wlog.curve_info(:,1);
    end
    
    log = getLog(labels, data, header);
    k(id) = length(data);
    
    % Add bitsize data (BS):
    [~,dataCO,headCO,~]=loadlas(compID{id});
    bs = getLog({'dept','bs','gr'}, dataCO, headCO); % depth and bs
    bs_new = (interp1(bs(:,1), bs(:,2), log(:,1),'Previous'));
    bs_new = bitFix(bs_new);
    
    % Add gamma ray (GR):
    gr = (interp1(bs(:,1), bs(:,3), log(:,1),'Previous'));
    gr = nanFix(gr);        % Fix interwell nan
    
    % stitch BS and GR: 
    log(:,end+1) = bs_new;
    log(:,end+1) = gr;
    
    
    t(1:length(log),:,id) = log;
end


labels{end + 1} = 'bs';
labels{end + 1} = 'gr';

end

