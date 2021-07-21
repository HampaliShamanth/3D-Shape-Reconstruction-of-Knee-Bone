function [probImage]=MultiLayerCanny_synthetic(Image)

probImage=double(Image*0);
pad=[0,0,2,0,1,2,5,0,2,6];
thresh=linspace(0.08,0.02,10);
weight=linspace(3,1,10);
startLevel=1; 
endLevel=7;

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
thresh2=sum(weight(endLevel-2:endLevel))/S;
probImage(probImage<=thresh2)=0;

%Remove small components
BW=bwareaopen(probImage>0,100);
probImage=probImage.*BW;
end