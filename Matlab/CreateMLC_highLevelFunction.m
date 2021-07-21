function [MLCframe]=CreateMLC_highLevelFunction(frame,borderMask,args,noise_array)
Mask=CreateMasks(frame,borderMask);
squishedFrame=SquishHistogram(frame);
Edgeimage=MultiLayerCanny(squishedFrame,args);
Edgeimage=Edgeimage.*Mask;

%imshow(Mask)

tiledNoise=Create_tile_for_Noise(noise_array,borderMask);
tiledNoise=tiledNoise.*(frame==0);
%imshow(Edgeimage)
MLCframe=Edgeimage+tiledNoise;
end