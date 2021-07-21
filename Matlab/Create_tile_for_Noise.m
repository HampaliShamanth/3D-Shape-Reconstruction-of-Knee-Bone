% Clean
% load NoiseCollection
% noiseTray=[];
% for i=1:size(NoiseCollection,2)
%     noiseTray=cat(1,noiseTray,NoiseCollection{1,i});
% end
% 
% posesPerImage=16;
% randomidx=randi([1 size(noiseTray,1)],posesPerImage,1);
% 
% noise_array=noiseTray(randomidx,:,:);
% 
% tiledNoise=Create_tile_for_Noise1(noise_array);
% imshow(tiledNoise);
function [tiledNoise]=Create_tile_for_Noise(noise_array,borderMask)
posesPerImage=size(noise_array,1);
rows=sqrt(posesPerImage);
cols=rows;

rowNoises=[];
tiledNoise=[];
idx=1;
for row=1:rows
    for col=1:cols
       rowNoises=[rowNoises,squeeze(noise_array(idx,:,:))]; 
   
      idx=idx+1;
    end
    tiledNoise=[tiledNoise;rowNoises];
    rowNoises=[];
end

tiledNoise=tiledNoise.*(borderMask/255);

end