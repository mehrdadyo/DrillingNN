
addpath('composite',genpath('S4M'),'MWD');

%TA Q:
% Should we standardize logs?

%% 
% This section reads logs from .las files missing Bitsize, then concatnates
% the BS log using interp1.
tic;
% Logs: for F1, F11, F11A, F11B, F11T2, F15D
hdrs = {'dept', 'ropavg_m_hr', 'wobavg_mton','surfrpm_rpm',...
    'torqueav_kj'};

mwdID = {'F1_MWD_CVTD.las', 'F1A_MWD_CVTD.las', 'F1B_MWD_CVTD.las',...
    'F1C_MWD_CVTD.las', 'F11_MWD_1.las','F11A_MWD_1.las', ...
    'F11B_MWD_1.las', 'F11T2_MWD_1.las', 'F15D_MWD_1.las'};

compID = {'F1_COMP.las', 'F1A_COMP.las', 'F1B_COMP.las', ...
    'F1c_COMP.las', 'F11_COMP.las','F11A_COMP.las', ...
    'F11B_COMP.las','F11T2_COMP.las', 'F15D_COMP.las'};

[logs1,~, loglen1] = mergeWell(mwdID,hdrs,compID);

% the following set of logs have different header labels...

% F4, F5, F7, F9, F9A, F10, F12, F14, F15, F15A, F15B, F15C
hdrs = {'dept','rop5','swob', 'rpm','tqa'};

mwdID = { 'F12_MWD_1.LAS','F12_MWD_2.LAS', 'F12_MWD_3.LAS', 'F12_MWD_4.LAS',...
    'F15_MWD_1.LAS','F15_MWD_2.LAS','F15_MWD_3.LAS',...
    'F4_MWD_1.LAS','F4_MWD_2.LAS','F4_MWD_3.LAS',...
    'F5_MWD_1.LAS','F5_MWD_2.LAS', 'F5_MWD_3.LAS',  ...
    'F10_MWD_1.LAS', 'F10_MWD_2.LAS', 'F10_MWD_3.LAS',...
    'F9A_MWD_1.LAS', 'F9A_MWD_2.LAS',...
    'F14_MWD_2.LAS', 'F14_MWD_3.LAS', ...
    'F15A_MWD_1.LAS', 'F15A_MWD_2.LAS', ...
    'F15C_MWD_1.LAS', 'F15C_MWD_2.LAS'};

compID = {'F12_COMP.las','F12_COMP.las', 'F12_COMP.las', 'F12_COMP.las',...
    'F15_COMP.las', 'F15_COMP.las','F15_COMP.las',...
    'F4_COMP.las', 'F4_COMP.las', 'F4_COMP.las', ...
    'F5_COMP.las','F5_COMP.las','F5_COMP.las',  ...
    'F10_COMP.las','F10_COMP.las', 'F10_COMP.las',...
    'F9A_COMP.las', 'F9A_COMP.las',...
    'F14_COMP.las', 'F14_COMP.las', ...
    'F15A_COMP.las', 'F15A_COMP.las', ...
    'F15C_COMP.las', 'F15C_COMP.las'};
[logs2,~, loglen2] = mergeWell(mwdID,hdrs,compID);

% F14_MWD_1:
hdrs ={'DEPTH','rop5','swob', 'rpm','tqa'};

mwdID = {'F7_MWD_1.las','F14_MWD_1.LAS','F9_MWD_1.LAS','F15B_MWD_1.LAS'};
compID = {'F7_COMP.las','F14_COMP.las','F9_COMP.las', 'F15B_COMP.las'};
[logs3,~, loglen3] = mergeWell(mwdID,hdrs,compID,0);

clear mwdID compID  hdrs ;

% Concatenate logs to be predicted

Logs = cat(3, logs1, logs2, logs3);
Hdrs = {'Depth', 'ROP', 'WOB', 'RPM', 'Torque', 'Bit Size', 'GR'};

readTime = toc;


%%
% Data Augmentation 
% The following window scans all logs with a specific STRIDE to generate
% log sets with height WINDOW.
tic;
window = 50;
stride = 10;
logSum = sum(loglen1)+sum(loglen2) + sum(loglen3);
n = ceil((logSum - window)/stride + 1);   % # of sets estimate

% Augmentation:
superSet = logAug(Logs,window,stride, n);

% Shuffle
[shuffledSet] = Shuffle(superSet, window);

check = sum(isnan(shuffledSet));

AugTime = toc;

clear loglen1 loglen2 loglen3;

%% Save to txt file:

fileID = fopen('shuffledSet.txt','w');
fprintf(fileID,'%10.4f  %10.4f  %10.4f  %10.4f  %10.4f  %10.4f  %10.4f\n'...
    ,shuffledSet');
fclose(fileID);

a = shuffledSet(:,:,1);

Hdrs =string(Hdrs);

fileID = fopen('set1_hdrs.txt','w');
fprintf(fileID,'%s    %s     %s     %s     %s     %s     %s'...
    ,Hdrs);
fclose(fileID);










%%
% Notes:
% there a difference in units of torque
% F9A: STOR was changed to TQA
% fix the 3e5 zeros
%
% idea: duplicate imporant logs

% moving forward without tvd, might not generalize our model regionally

% QC: reference datum for wells
% deal with nans: regression, look for a better log




scatter (logs3(:,end), logs3(:,3).*logs3(:,5))

%%
a = [1 3 6];
b = [2 6 10];

m1 = mean(a);
s1 = std(a);
m3 = mean([m1 m2]);

s3 = sqrt(s1.^2 + s2.^2);
m2 = mean(b);
s2 = std(b);

c = [1 2 3 6 6 10];
mt = mean(c);
st = std(c);








