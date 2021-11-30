Classes = {'background', 'Spartina', 'Dead Spartina', 'Sarcocornia', 'Batis', 'Juncus', 'Borrichia', 'Limoninum'};
ClassColorsList= uint8([255, 255, 255;  % background
                        127, 255, 140;  % Spartina
                        113, 255, 221;  % dead Spartina
                        99, 187, 255;   % Sarcocornia
                        101, 85, 255;   % Batis
                        212, 70, 255;   % Juncus
                        255, 56, 169;  % Borrichia
                        255, 63, 42]);% Limonium
    
    
figure
ClassColorsPlot = reshape(ClassColorsList, 8, 1, 3);
imagesc(ClassColorsPlot);
set(gca,'XTick',[]); set(gca,'YTick',[]);
imLabel(Classes,'right');