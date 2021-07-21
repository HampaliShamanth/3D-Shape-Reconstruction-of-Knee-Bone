function [MLC_frame]=AddNoise_and_CreateMask(frame)

posesperImage=16;

%% Mask to remove bone caps
Y=frame/max(frame(:));
Y(Y>0.03)=0;
Y(Y>0)=1;
Y=medfilt2(medfilt2(Y));
Y=bwareaopen(Y,10);
boneCapMask=imcomplement(Y);


%% Mask for tile borders
rows=sqrt(posesperImage);
cols=rows;
r=256;  %TileSize

borderMask=zeros(r*rows, r*cols);

stripes=0:r:r*rows;
stripes(1)=1;

borderMask(stripes,:)=1;
borderMask(:,stripes)=1;

borderMask=borderMask>0;
SE = strel('square', 15);
borderMask=imdilate(borderMask,SE);


borderMask=imcomplement(borderMask);

%% Aura Mask
SE = strel("disk",1);
auraMask = imdilate((frame>0),SE);

%% Histogram squishing
LL=0.1;
UL=1;
squished_frame = imadjust(frame/max(frame(:)),[0.01 1],[LL UL],(rand(1)+1));
squished_frame(squished_frame==LL)=0;

%% Creating a max value so that canny thresh works
squished_frame(:,1024)=1;

%% Adding Noise
BW=double(imcomplement(squished_frame>0));
BW(BW==0)=0.1;
Noise=imnoise(BW*0,'gaussian',LL-0.02,0.2).*BW;

%squished_frame=imnoise(squished_frame,'salt & pepper',0.05);
squished_frame=squished_frame+Noise;
%imgaussfilt(squished_frame+Noise,0.1);
squished_frame=imnoise(squished_frame,'speckle',0.005);
%imshow(squished_frame)
%% Mask
Mask=auraMask.*borderMask.*boneCapMask;

%% Create MultiLayer Canny
args.start=1;
args.end=6;
args.weight=3;

MLC_frame=MultiLayerCanny(squished_frame,args);
MLC_frame=MLC_frame.*auraMask.*borderMask.*boneCapMask;

end