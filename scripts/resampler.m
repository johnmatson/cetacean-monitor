orca_files=dir(fullfile('datasets/best/','*.wav'))

for k=1:numel(orca_files)
    workingFile=orca_files(k).name; %pull up current working file
    workingFileLocation=append('datasets/best/',workingFile); % append to file location
    [y,Fs] = audioread(workingFileLocation);
    y = resample(y,16000,Fs);
    name = sprintf('datasets/best-resampled/%s',orca_files(k).name);
    audiowrite(name,y,16000);
end