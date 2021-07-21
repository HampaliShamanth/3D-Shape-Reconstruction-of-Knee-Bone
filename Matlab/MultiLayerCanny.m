function [probImage]=MultiLayerCanny(Image,args)
Image=medfilt2(medfilt2(Image)); %Median filter

probImage=double(Image*0);
pad=[0,0,2,0,1,2,5,0,2,6,10,8,3,12,11];

if size(Image,1)==256
    pad=[0,0,2,0,4,2,3,0,5,4,8,8,4,0,14];
end

startLevel=args.start;
endLevel=args.end;
thresh=linspace(0.06,0.02,endLevel);
weight=linspace(args.weight,1,endLevel);

for i=startLevel:endLevel
    n=pad(i);
    I=padarray(Image,[n,n],'pre'); % pad
    I =imresize(I,1/i);
    %     I=imnoise(I,'gaussian',0.01,0.0001);
    %     I=adapthisteq(I);
    %     I=imresize( I,i,'bicubic');
    %     I =  I(n+1:end,n+1:end); % unpad
    edgeImage = edge(I,'Canny','nothinning',thresh(i));
    edgeImage=imresize(edgeImage,i,'bicubic');
    edgeImage = edgeImage(n+1:end,n+1:end); % unpad
    probImage = double(edgeImage)*weight(i)+probImage;
end
S=sum(weight(startLevel:endLevel));
probImage=probImage/S;
thresh2=sum(weight(endLevel-1:endLevel))/S;
probImage(probImage<=thresh2)=0;

%Remove small components
BW=bwareaopen(probImage>0,100);
probImage=probImage.*BW;
end