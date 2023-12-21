function plottopomap(elocsX,elocsY,elabels,data)

% define XY points for interpolation
interp_detail = 100;
interpX = linspace(min(elocsX)-.2,max(elocsX)+.25,interp_detail);
interpY = linspace(min(elocsY),max(elocsY),interp_detail);

% meshgrid is a function that creates 2D grid locations based on 1D inputs
[gridX,gridY] = meshgrid(interpX,interpY);
% Interpolate the data on a 2D grid
interpFunction = TriScatteredInterp(elocsY,elocsX,data);
topodata = interpFunction(gridX,gridY);

% plot map
contourf(interpY,interpX,topodata);
hold on
scatter(elocsY,elocsX,10,'ro','filled');
for i=1:length(elocsX)
    text(elocsY(i),elocsX(i),elabels(i))
end
set(gca,'xtick',[])
set(gca,'ytick',[])
