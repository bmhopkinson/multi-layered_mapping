addpath(genpath('/Users/brianhopkinson/Dropbox/MATLAB/Library/matGeom'));  %% matalb geom3d package by David Legland - load off 

base_dir = './Sapelo_202106_run13/';
meshFile = strcat(base_dir,'mesh_fine.ply');
%face_color_file = strcat(base_dir,'run_13_face_classcolors_20210909_model.txt');
face_color_file = strcat(base_dir,'run_13_face_truecolors.txt');

[V, F] = readMesh_ply(meshFile);
n_faces = size(F,1);
face_colors = zeros(n_faces,3);

fin = fopen(face_color_file);

C = textscan(fin, '%d\t%d\t%d\t%d\n');

face_ids = C{1};
%[face_ids, idx] = sort(face_ids);

%handle face colors
R = double(C{2})/255.0;
G = double(C{3})/255.0;
B = double(C{4})/255.0;
obs_colors =[R G B];
%R = R(idx)/255;
%B = B(idx)/255;
%G = G(idx)/255;

face_colors(face_ids,:) = obs_colors;

%mesh often extends beyond labeled portion, find extents of labelled
%portion for setting axis limits
v_ann_idxs = F(face_ids,:);
v_ann_idxs = unique(v_ann_idxs(:));
V_ann = V(v_ann_idxs,:);
v_ann_min = min(V_ann, [],1);
v_ann_max = max(V_ann, [],1);

figure
%colormap(MyColorMap);
surf_plot = trisurf(F, V(:,1), V(:,2), V(:,3));
surf_plot.FaceVertexCData = face_colors;
xlim([v_ann_min(1) v_ann_max(1)]);
ylim([v_ann_min(2) v_ann_max(2)]);
zlim([v_ann_min(3) v_ann_max(3)]);
xlabel('meters');
ylabel('meters');
zlabel('meters');
shading flat;
%axis image;

%camlight(150,90);