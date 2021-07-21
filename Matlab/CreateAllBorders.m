function [borderMask]=CreateAllBorders(posesPerImage,margin)
rows=sqrt(posesPerImage);
cols=rows;
r=256;  %TileSize

borderMask=zeros(r*rows, r*cols);


stripes=0:r:r*rows;
stripes(1)=1;

borderMask(stripes,:)=1; %Somehigh number
borderMask(:,stripes)=1; %Somehigh number

borderMask=borderMask>0;
SE = strel('square', margin);
borderMask=imdilate(borderMask,SE);


%borderMask=im2uint16(borderMask);

end
