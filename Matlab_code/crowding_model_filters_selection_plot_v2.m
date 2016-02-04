% function [] = crowding_model_filters()

stim_size = 101;
center_location = ceil(stim_size/2);

RF_horiz_size = 50;
RF_vert_size = 50;
RF_scaling_slope = .75;

num_RFs = 24; %divisible by 3
num_training_stims = 120; %divisible by 3
num_test_stims = 60; %divisible by 3
num_all_stims = num_training_stims + num_test_stims;

num_its = 50;

load('C:\Users\Anna\Dropbox\Crowding_Model_Materials\Matlab_code\data_files\filters.mat','filters'); %load filters

range_data = zeros(2,19);

for drop_range = [.7 .7 .7 .7]%.05:.05:.95
disp('starting a new set')
all_percent_correct = zeros(num_its,1);
all_sparse_percent_correct = zeros(num_its,1);

for iteration = 1:num_its

    
    stim_size = 101;
    center_location = ceil(stim_size/2);

    RF_horiz_size = 50;
    RF_vert_size = 50;
    RF_scaling_slope = .75;

    num_RFs = 24; %divisible by 3
    num_training_stims = 120; %divisible by 3
    num_test_stims = 60; %divisible by 3
    num_all_stims = num_training_stims + num_test_stims;
    
    %generate RFs
    % x location | y location | tuning
    RFs = [randi(stim_size,num_RFs,2)-center_location [ones(num_RFs/3,1); 2*ones(num_RFs/3,1); 3*ones(num_RFs/3,1)]];

    RF_images = zeros(stim_size,stim_size,num_RFs);
    [X,Y] = meshgrid(-(center_location-1):(center_location-1),-(center_location-1):(center_location-1));

    for RF_num = 1:num_RFs
        horiz_size = (RF_horiz_size/2)*(1+RF_scaling_slope*RFs(RF_num,1)/center_location);
        vert_size = (RF_vert_size/2)*(1+RF_scaling_slope*RFs(RF_num,1)/center_location);
        RF_images(:,:,RF_num) = ((X-RFs(RF_num,1))/horiz_size).^2 + ((Y-RFs(RF_num,2))/vert_size).^2 <= 1;
    end


    % training & test stimuli

    flanker_positions = (10:stim_size-10);
    limited_flanker_positions = [(10:center_location-20) (center_location+20:stim_size-10)];
    training_flanker_data = zeros(num_all_stims,6);

    training_stims = zeros(stim_size,stim_size,num_all_stims); %this will include test stims, then they're pulled out
    for stim_num = 1:num_all_stims
        stim_image = zeros(stim_size,stim_size);
        stim_image(center_location-9:center_location+10,center_location-9:center_location+10) = ...
            stim_image(center_location-9:center_location+10,center_location-9:center_location+10) + ...
            filters(:,:,ceil(3*stim_num/num_all_stims)); %place target at center

        %flanker #1
        flanker_location = flanker_positions(randi(length(flanker_positions)));
        if flanker_location > center_location - 20 && flanker_location < center_location + 20
            flanker_location(2) = limited_flanker_positions(randi(length(limited_flanker_positions)));
        else
            flanker_location(2) = flanker_positions(randi(length(flanker_positions)));
        end

        flanker_location = flanker_location(randperm(2)); %randomize (x,y) order 

        flanker_type = randi(3);
        training_flanker_data(stim_num,1:3) = [flanker_location flanker_type];

        stim_image(flanker_location(1)-9:flanker_location(1)+10,flanker_location(2)-9:flanker_location(2)+10) = ...
            stim_image(flanker_location(1)-9:flanker_location(1)+10,flanker_location(2)-9:flanker_location(2)+10) + ...
            filters(:,:,flanker_type); %randomly place one flanker

        %flanker #2
        flanker_location = flanker_positions(randi(length(flanker_positions)));
        if flanker_location > center_location - 20 && flanker_location < center_location + 20
            flanker_location(2) = limited_flanker_positions(randi(length(limited_flanker_positions)));
        else
            flanker_location(2) = flanker_positions(randi(length(flanker_positions)));
        end

        flanker_location = flanker_location(randperm(2)); %randomize (x,y) order 

        flanker_type = randi(3);
        training_flanker_data(stim_num,4:6) = [flanker_location flanker_type];

        stim_image(flanker_location(1)-9:flanker_location(1)+10,flanker_location(2)-9:flanker_location(2)+10) = ...
            stim_image(flanker_location(1)-9:flanker_location(1)+10,flanker_location(2)-9:flanker_location(2)+10) + ...
            filters(:,:,flanker_type); %randomly place one flanker

        stim_image(stim_image > 1) = 1;

        training_stims(:,:,stim_num) = stim_image;
    end

    
    %generate vector of RF outputs
    RF_responses = zeros(num_RFs,size(training_stims,3));
    for stim_num = 1:size(training_stims,3)
        current_stim = training_stims(:,:,stim_num);
        filtered_stim = cat(3,normxcorr2e(filters(:,:,1),current_stim,'same'),normxcorr2e(filters(:,:,2),current_stim,'same'),normxcorr2e(filters(:,:,3),current_stim,'same'));

        for RF_num = 1:num_RFs

            RF_pooled = RF_images(:,:,RF_num).*filtered_stim(:,:,RFs(RF_num,3)); %pool over RF using correct filter
            RF_response = max(max(RF_pooled));
            
%             if RF_response < .75 %discrete response
%                 RF_response = 0;
%             else
%                 RF_response = 1;
%             end

            if RF_response < 0     %graded response
                RF_response = 0;
            end

            RF_responses(RF_num,stim_num) = RF_response;
        end
    end
    
    %divide training and test data
    test_indices = [(1:num_test_stims/3) (num_all_stims/3 + 1:num_all_stims/3 + num_test_stims/3) (2*num_all_stims/3 + 1:2*num_all_stims/3 + num_test_stims/3)];
    test_stims = training_stims(:,:,test_indices);
    training_stims(:,:,test_indices) = []; %remove test stimuli from training data

    training_RF_responses = RF_responses;
    test_RF_responses = training_RF_responses(:,test_indices);
    training_RF_responses(:,test_indices) = []; %remove test stimuli from training data


    training_targets = [repmat([1 0 0]',[1 num_training_stims/3]) repmat([0 1 0]',[1 num_training_stims/3]) repmat([0 0 1]',[1 num_training_stims/3])];
    %shuffle training targets for null distribution
%     training_targets = training_targets(:,randperm(size(training_targets,2)));
%     
    
    test_targets = [repmat([1 0 0]',[1 num_test_stims/3]) repmat([0 1 0]',[1 num_test_stims/3]) repmat([0 0 1]',[1 num_test_stims/3])];

    %train network
    net = patternnet(10);
    net.trainParam.showWindow = false;
    net.trainParam.showCommandLine = false;
    net = train(net,training_RF_responses,training_targets);

    %test network
    predictions = net(test_RF_responses);
    prediction_classes = vec2ind(predictions);
    percent_correct = mean(prediction_classes == vec2ind(test_targets));
    disp(percent_correct)
    all_percent_correct(iteration) = percent_correct;
    
    %take out some inputs
    input_weights = net.IW;
    proportion_drop = drop_range;
    input_weights{1}(:,randsample(1:size(input_weights{1},2),ceil(proportion_drop*size(input_weights{1},2)))) = 0;
    net.IW = input_weights;
    
    %original test set
    predictions = net(test_RF_responses);
    prediction_classes = vec2ind(predictions);
    percent_correct = mean(prediction_classes == vec2ind(test_targets));
    disp(percent_correct)
    all_sparse_percent_correct(iteration) = percent_correct;
    clearvars -except filters all_percent_correct all_sparse_percent_correct range_data num_its drop_range
end %end iterations

range_data(1,round(drop_range*20)) = mean(all_percent_correct);
range_data(2,round(drop_range*20)) = mean(all_sparse_percent_correct);

%save('/Users/jasonfischer/Dropbox/MATLAB/crowding_model/range_data_1000.mat','range_data')

end %end drop range





% end %end main function