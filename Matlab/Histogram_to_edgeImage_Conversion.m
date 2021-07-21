clear all;
close all;
clc;

inputPath="D:\OneDrive - University of Waterloo\Thesis\Projects\IC\Synthetic Images\Histograms\Set 1";
outputPath="D:\OneDrive - University of Waterloo\Thesis\Projects\IC\Synthetic Images\EdgeImages\Set 2";
cd(inputPath)

shapeParamsOffset=10;%From python file
multiplicationFactor = 10;
tiles=16;

inputDir=dir(inputPath);
inputDir(ismember( {inputDir.name}, {'.', '..','Index.xlsx'})) = [];

outputDir=dir(outputPath);
outputDir(ismember( {outputDir.name}, {'.', '..','Index.xlsx'})) = [];

filesToBeConverted=size(inputDir,1)-size(outputDir,1);

files = dir(fullfile(cd, '*.png'));
[~,T_idx]=natsortfiles({files.name});  %Name sorting
files=files(T_idx);
i=1;
for i=size(outputDir,1)+1:size(inputDir,1)
    
    inputImage=double(imread(files(i).name));
    %Extract shape parameters
    shapeParams=inputImage(1:49,1025);
    shapeParams=(shapeParams/1000)-shapeParamsOffset;
    Coordinates=GetKneeCoordinates(shapeParams);
    Coordinates=round((Coordinates+1000)*10);
    %%
    %Preprocess the image
    X=inputImage/multiplicationFactor;
    X=X(:,1:1024);
    
    [squished_frame,Mask]=AddNoise_and_CreateMask(X);
    
    args.start=1;
    args.end=6;
    args.weight=3;
    MLC_frame=MultiLayerCanny(squished_frame,args);
    [MLC_frame]=CreateSyntheticEdgeImage(X);    %MultiLayerCanny Frame
    
    MLC_frame=MLC_frame.*Mask;
    
    imshow(Mask);
    
    reminder=1024-rem(size(Coordinates,1),1024);
    dummy=zeros(reminder,1);
    
    MLC_frame=[MLC_frame(:)*1000;Coordinates;dummy];
    MLC_frame=reshape(MLC_frame,1024,[]);
    
    fileName=join([outputPath,files(i).name],"");
    imwrite(uint16(MLC_frame),fileName);
    subplot(1,2,1)
    imshow(im2uint8(J));
    imagesc(X);colormap gray
    subplot(1,2,2)
    imagesc(edgeImage);colormap gray
    
end