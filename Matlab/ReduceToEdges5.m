function [edgeImage,highContrastImage]=ReduceToEdges5(Image,filter,display)
if ~exist('display','var') display=0; end

level=filter.sigma2;
Gsigma=filter.Gsigma;
thresh=filter.thresh;
P=filter.P;
cL=filter.clipLimit;
dist=filter.Distribution;
filterName=filter.name;

Image=medfilt2(medfilt2(Image)); %Median filter

% [probImage]=MultiLayerCanny(Image);
% 
% imshow((probImage));
% imagesc(probImage); colormap gray;
% A=im2uint8(probImage);

pad=[0,0,2,0,1,2,5,0,2,6];
if size(Image,1)==256
    pad=[0,0,2,0,4,2,3,0,5,4,8,8,4,0,14];
end
n=pad(level);

Image=padarray(Image,[n,n],'pre'); % pad
Image =imresize(Image,1/level,'Antialiasing',true);

% Image=imresize(Image,level,'bilinear');
% Image = Image(n+1:end,n+1:end); % unpad
% imshow(Image)

switch filterName
    case "GradientI"
        [edgeImage,~]=imgradient(Image);
    case "Canny"
        %Image=imadjust(Image);
        edgeImage = edge(Image,filterName,'nothinning',thresh);
        edgeImage=bwareaopen(edgeImage,P*2);       
    case "Sobel"        
        edgeImage=0*Image;
end
edgeImage=imresize(edgeImage,level,'bicubic','Antialiasing',true);
edgeImage = edgeImage(n+1:end,n+1:end); % unpad

if display
    if ~ishandle(2)
        figure(2);
    end
    figure(2);
    %     subplot(1,2,1)
    %     imshow(imadjust(Image,[0.0,0.2],[]));
    %     %     subplot(1,3,2)
    %     %     imshow(BiImage);
    %     subplot(1,2,2)
    %     imshow(imadjust(highContrastImage));
    H=(imadjust(im2uint8(Image)));
    subplot(1,2,1)
    imhist(double(H)/255)
    ylim([0 20000]);
    subplot(1,2,2)
    imshow(H);
    
end


end