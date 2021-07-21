function [sortedFiles]=Sortfiles(files)
[~,T_idx]=natsortfiles({files.name});  %Name sorting
sortedFiles=files(T_idx);
end