addpath(genpath('/Users/brianhopkinson/Dropbox/MATLAB/Library/matGeom'));  %% matalb geom3d package by David Legland - load off 
addpath(genpath('/Users/brianhopkinson/Dropbox/MATLAB/Computer_Vision/pdollar_toolbox'));      %pdollar toolbox - various supporting functions - here imLabel

plot_plant = 'Sarcocornia';

base_dir = './Sapelo_202106_run13/';
meshFile = strcat(base_dir,'mesh_fine.ply');
classifiedFile = strcat(base_dir,'run_13_fractional_cover_by_face_fine.txt');

plants = {'background', 'Spartina', 'Dead_Spartina', 'Sarcocornia', 'Batis', 'Juncus', 'Borrichia', 'Limoninum'};
plant_idx_cover = [2, 3, 4, 5, 6, 7, 8, 9];
plant_to_idx_cover = containers.Map(plants, plant_idx_cover);

plant_idx_color = [1, 2, 3, 4, 5, 6, 7, 8];
plant_to_idx_color = containers.Map(plants, plant_idx_color);

ClassColorsList= [255, 255, 255;  % background
                  127, 255, 140;  % Spartina
                  113, 255, 221;  % dead Spartina
                   99, 187, 255;   % Sarcocornia
                  101, 85, 255;   % Batis
                  212, 70, 255;   % Juncus
                  255, 56, 169;  % Borrichia
                  255, 63, 42];% Limonium
    
[V, F] = readMesh_ply(meshFile);

fin = fopen(classifiedFile);

C = textscan(fin, '%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n');

f_label = C{1};
plant_cover = C{plant_to_idx_cover(plot_plant)};
n_faces = size(F,1);

face_colors = zeros(n_faces,1);
face_colors(f_label) = plant_cover;

%mesh often extends beyond labeled portion, find extents of labelled
%portion for setting axis limits
v_ann_idxs = F(f_label,:);
v_ann_idxs = unique(v_ann_idxs(:));
V_ann = V(v_ann_idxs,:);
v_ann_min = min(V_ann, [],1);
v_ann_max = max(V_ann, [],1);

%colormaps
res = 100;
g = linspace(0, 1,res)';
end_member1 = [1.0, 1.0, 1.0]; %white
end_member2 = double(ClassColorsList(plant_to_idx_color(plot_plant),:))./255.0; 
custom_cmap = (1-g)*end_member1 + g*end_member2;   
colormap(custom_cmap);

trisurf(F, V(:,1), V(:,2), V(:,3), face_colors);
xlim([v_ann_min(1) v_ann_max(1)]);
ylim([v_ann_min(2) v_ann_max(2)]);
zlim([v_ann_min(3) v_ann_max(3)]);
xlabel('meters');
ylabel('meters');
zlabel('meters');
shading flat;

%camlight(150,90);