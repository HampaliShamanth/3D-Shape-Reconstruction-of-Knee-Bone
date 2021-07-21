function [squishedFrame]=SquishHistogram(frame)
%% Histogram squishing
UL=0.8;
LL=0.55+rand(1)*0.15;
%LL=0.55- More information
%LL=0.75- Less information

squishedFrame = imadjust(frame/max(frame(:)),[0.01 1],[LL UL],(1));%rand(1)*3));
squishedFrame(squishedFrame==LL)=0;

%% Creating a max value so that canny thresh works
squishedFrame(:,1024)=1;

%% Adding Noise
BW=double(imcomplement(squishedFrame>0));

BW(BW==0)=0.1;
Noise=imnoise(BW*0,'gaussian',LL-0.02,0.001).*BW;
squishedFrame=squishedFrame+Noise;

%squishedFrame=imnoise(squishedFrame,'salt & pepper',0.05);
%imgaussfilt(squishedFrame+Noise,0.1);
%squishedFrame=imnoise(squishedFrame,'speckle',0.005);
%imshow(squishedFrame)


end