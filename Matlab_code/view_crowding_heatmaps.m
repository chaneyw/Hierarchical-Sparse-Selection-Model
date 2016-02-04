mat_filenames = uipickfiles('FilterSpec','/Users/jasonfischer/Dropbox/MATLAB/crowding_model/*.mat','Prompt','Select .mat files:','NumFiles',[],'Output','cell');

crowding_heatmap = zeros(91, 91);

for file_num = 1:size(mat_filenames,2)
    load(mat_filenames{file_num})
    overall_performance_image(overall_performance_image == 0) = [];
    crowding_heatmap = crowding_heatmap + reshape(overall_performance_image,[91 91]);
end

figure
imagesc(crowding_heatmap)
