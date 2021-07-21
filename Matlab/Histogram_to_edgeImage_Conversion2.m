clear all;
close all;
clc;

inputPath="D:\OneDrive - University of Waterloo\Thesis\Projects\IC\Python\ML1\Code\Synthetic Images";
outputPath="D:\OneDrive - University of Waterloo\Thesis\Projects\IC\Python\ML1\Code\MultilayerCanny Images";
cd(inputPath)

%From python files  (functions.py> def TileImages())
shapeParamsOffset=10;%From python file
image_multiplicationFactor = 10;
shape_amp=1000;%# shape parameters amplifying factor


posesPerImage=16;

inputDir=dir(inputPath);
inputDir(ismember( {inputDir.name}, {'.', '..','Index.xlsx'})) = [];
inputDir=Sortfiles(inputDir);

outputDir=dir(outputPath);
outputDir(ismember( {outputDir.name}, {'.', '..','Index.xlsx'})) = [];
outputDir=Sortfiles(outputDir);

args.start=1;
args.end=5;
args.weight=2;

borderMask=CreateAllBorders(posesPerImage,15);
borderMask=double(imcomplement( borderMask>0));
load NoiseCollection
noiseTray=[];
for i=1:size(NoiseCollection,2)
    noiseTray=cat(1, noiseTray,NoiseCollection{1,i});
end
noiseTray=double(noiseTray);

for folder=1:size(inputDir,1)
    cd(inputPath)
    cd(inputDir(folder).name)
    sprintf('Folder#%d',folder)
    files = dir(fullfile(cd, '*.png'));
    if size(files,1)~=100
        continue
    else
        files=Sortfiles(files);
    end
    
    %Check if the last file in the folder is converted to MLC
    lastFile=strcat(num2str(folder*100),'.png');
    if any(ismember({outputDir.name},{lastFile}))
        continue;
    end
    
    for j=1:size(files,1)
        inputImage=double(imread(files(j).name));
        
        %Extract shape parameters
        shapeParams=inputImage(1:49,1025,1);
        shapeParams=(shapeParams/shape_amp)-shapeParamsOffset;
        Coordinates=GetKneeCoordinates(shapeParams);
        Coordinates=round((Coordinates+1000)*10);
        
        reminder=1024-rem(size(Coordinates,1),1024);
        dummy=zeros(reminder,1);
        coordinatesSpace=reshape([Coordinates;dummy],1024,[]);
        A=repmat(coordinatesSpace,[1,1,3]);
        
        %Preprocess the image
        X=inputImage/image_multiplicationFactor;
        X=X(:,1:1024,:);
        for channel=1:size(X,3)
            frame=X(:,:,channel);

            randomidx=randi([1 size(noiseTray,1)],posesPerImage,1);
            noise_array=noiseTray(randomidx,:,:);
       
            
            MLCframe(:,:,channel)=CreateMLC_highLevelFunction(frame,borderMask,args,noise_array);
            
        end
        if max(MLCframe(:)>2.0)
            print("Oops");
            pause;
        end
        
        
        finalImage=cat(2,MLCframe*30000,A);
        
        imageNum=join([num2str((folder-1)*100+j),'.png'],"");
        fileName=join([outputPath,imageNum],"\");
        imwrite(uint16(finalImage),fileName);
    end
    %     subplot(1,2,1)
    %     imshow(im2uint8(finalImage));
    %     imagesc(X);colormap gray
    %     subplot(1,2,2)
    %     imagesc(edgeImage);colormap gray
    
    
    
end

