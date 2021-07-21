function [Mask]=CreateMasks(frame,borderMask)

posesperImage=16;

%% Mask to remove bone caps
Y=frame/max(frame(:));
Y(Y>0.03)=0;
Y(Y>0)=1;
SE = strel("disk",1);
Y=imopen(Y,SE);
% figure(3);
% imshow(Y);
boneCapMask=imcomplement(Y);
SE = strel("rectangle",[5,70]);
boneCapMask = imerode(boneCapMask,SE);


% %% Mask for tile borders
% rows=sqrt(posesperImage);
% cols=rows;
% r=256;  %TileSize
% 
% borderMask=zeros(r*rows, r*cols);
% 
% stripes=0:r:r*rows;
% stripes(1)=1;
% 
% borderMask(stripes,:)=1;
% borderMask(:,stripes)=1;
% 
% borderMask=borderMask>0;
% SE = strel('square', 15);
% borderMask=imdilate(borderMask,SE);
% 
% 
% borderMask=imcomplement(borderMask);

%% Aura Mask
SE = strel("disk",1);
auraMask = imdilate((frame>0),SE);

%% Mask
%Mask=auraMask.*borderMask.*boneCapMask;
Mask=auraMask.*boneCapMask.*borderMask;
Mask=Mask>0;

end