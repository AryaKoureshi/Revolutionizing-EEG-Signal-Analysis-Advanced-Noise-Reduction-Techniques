% 32-channel data
load('Electrodes') ;

% Plot Data
% Use function disp_eeg(X,offset,feq,ElecName)
offset = max(abs(X(:))) ;
feq = 200 ;
% ElecName = Electrodes.labels ;
disp_eeg(X,offset,feq,[]);