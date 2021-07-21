clear all;
close all;
clc;

for folder=1:10
    Create_tile_for_real_images(folder)
end

function Create_tile_for_real_images(folder)

cd 'D:\OneDrive - University of Waterloo\Thesis\Projects\IC\Datasets\Fluoroscopy Knee brace Study\Knee brace study\Stage3_Subjects';
cd(num2str(folder))
X = squeeze(dicomread("Rotation"));
frames=squeeze(num2cell(X,[1,2]));

load rectPositions.mat
load circleMask.mat

load markers.mat
M=markers.("Rotation");
totalFrames=size(X,3);
for i=1:size(M,2)
    M(:,i)=smooth(M(:,i));
end
Tform=FindTform2( M(1:totalFrames,[1:4]));

boundingBox=rectPositions.("Rotation");


if (folder==1)
    X = squeeze(dicomread("Rotation2"));
    frames2=squeeze(num2cell(X,[1,2]));
    frames=[frames2;frames];
    clear frames2
    %
    M=markers.("Rotation2");
    totalFrames=size(X,3);
    for i=1:size(M,2)
        M(:,i)=smooth(M(:,i));
    end
    Tform2=FindTform2( M(1:totalFrames,[1:4]));
    Tform=[Tform2, Tform];
    clear Tform2
    %
    boundingBox=[rectPositions.("Rotation2");rectPositions.("Rotation")];
    %Remove last two entries
    boundingBox(end,:)=[];
    boundingBox(end,:)=[];
end

[posesInaTile,N]=FindPosesInaTile(folder);

posesperImage=size(posesInaTile,2);

tiledPoses{posesperImage}=[];
tiledcirleMasks{posesperImage}=[];
for i=1:posesperImage
    k=1;
    for j=1:size(posesInaTile{i},1)
        frameNumber=posesInaTile{i}(j);
        if any(boundingBox(frameNumber,[1,2])~=[0,0])
            [T,C]=PreprocessImage(frameNumber,frames,...
                boundingBox,circleMask,Tform );
            tiledPoses{i}(:,:,k)=T;
            tiledcirleMasks{i}(:,:,k)=C;
            k=k+1;
        end
    end
    
end

save ("processedImages",'tiledPoses','tiledcirleMasks');

end

function [processedImage,circleMask]=PreprocessImage(frameNumber,frames, boundingBox,circleMask,Tform )
frame=frames{frameNumber};
frame=imwarp(frame,Tform(frameNumber),'OutputView',...
    imref2d(size(frame)),'FillValues', 0);
frame=imcrop(frame,boundingBox(frameNumber,:));
processedImage=imresize(frame,[256,256]);

circleMask=imwarp(circleMask,Tform(frameNumber),'OutputView',...
    imref2d(size(circleMask)),'FillValues', 0);
circleMask=imcrop(circleMask,boundingBox(frameNumber,:));
circleMask=imresize(circleMask,[256,256]);

end






