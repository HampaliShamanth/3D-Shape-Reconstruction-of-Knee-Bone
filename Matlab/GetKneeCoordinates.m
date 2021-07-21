function [Coordinates,fem,tib,pat,acl,pcl,mcl,lcl,pl]=GetKneeCoordinates(shapeParams)

MU=dlmread('PC_MU.dat');
COEFF=dlmread('PC_COEFFS.dat');
PC_DATA=dlmread('PC_DATA.dat');  %contains means and std devs from the training set


mu_pcs=PC_DATA(1,:);
sigma_pcs=PC_DATA(2,:);

% instance=dlmread('TRAINING_SET_INSTANCES.dat');
% shapeParams=instance(:,1);

Npc=size(shapeParams,1); % Number of PC scores
% Generate PC scores for specific instance
NewPCs=mu_pcs' + shapeParams.*sigma_pcs';

% Multiply from PC space to Cartesian space
NewData=COEFF'*NewPCs;

% Realign data with respect to the mean specimen
X=NewData+MU';

% Create data structure of the new instance
[S]=BuildModel_OS(X,1);

fem=S.bone.fem.coord.model(:,2:end);
tib=S.bone.tib.coord.model(:,2:end);
pat=S.bone.pat.coord.model(:,2:end);
acl=S.ligament.ACL.nodes.spg(:,2:end);
pcl=S.ligament.PCL.nodes.spg(:,2:end);
mcl=S.ligament.MCL.nodes.mem(:,2:end);
lcl=S.ligament.LCL.nodes.mem(:,2:end);
pl=S.ligament.PL.nodes.mem(:,2:end);

Coordinates=[fem;tib;pat;acl;pcl;mcl;lcl;pl];

Coordinates=Coordinates(:);
end