function [posesInaTile,N]=FindPosesInaTile(folder)

posesPerImage=16;
A=xlsread('Poses');

x=A(:,folder*2-1);
v=A(:,folder*2);

%Interpoalte and find missing pose value
x = x(~isnan(x));
v = v(~isnan(v));
xq=[min(x):max(x)]';
vq1 = interp1(x,v,xq);

%Categorise the poses into 16 bins
alphaInterval=(180-0)/(2*posesPerImage);
alpha=linspace(0,180,16);
edges=alpha-alphaInterval;
edges(end+1)=edges(end)+2*alphaInterval; %Range for each bin

[N,~,bin] = histcounts(vq1,edges);


for tileIndex=1:posesPerImage
    posesInaTile{tileIndex}= xq(bin==tileIndex);
end



end