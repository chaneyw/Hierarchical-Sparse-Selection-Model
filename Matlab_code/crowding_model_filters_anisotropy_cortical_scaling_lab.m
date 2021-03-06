% function [] = crowding_model_filters()

stim_size = 101;
center_location = ceil(stim_size/2);
num_its = 1000;

RF_horiz_size = 20; % was 50

RF_scaling_slope = .75;% .75;

num_RFs = 50; %divisible by 3
num_training_stims = 120;%120; %divisible by 3
num_test_stims = 60; %divisible by 3
num_all_stims = num_training_stims + num_test_stims;

%create filters
filters = zeros(20,20,2); 
%filters(:,:,1) = [zeros(1,20); zeros(1,20); repmat([0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0],[16,1]); zeros(1,20); zeros(1,20)];
filters(:,:,1) = repmat([0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0],[20,1]);
filters(:,:,2) = filters(:,:,1)';

%this creates variable stimuli
%load('C:\Users\Anna\Dropbox\Crowding_Model_Materials\Matlab_code\data_files\stimuli2.mat','stimuli');
stimuli = zeros(20,20,4);
stimuli(:,:,1) = [zeros(1,20); zeros(1,20); [0 0 0 0 ones(1,12) 0 0 0 0]; [0 0 0 0 ones(1,12) 0 0 0 0]; repmat([0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0],[14,1]); zeros(1,20); zeros(1,20)];
stimuli(:,:,2) = flipud(stimuli(:,:,1));
stimuli(:,:,3) = stimuli(:,:,1)';
stimuli(:,:,4) = fliplr(stimuli(:,:,3));

save_name = input('Name of save file: ','s');

all_percent_correct = zeros(num_its);
all_sparse_percent_correct = zeros(num_its,6);

all_test2_sparse_percent_correct_leftright = zeros(num_its,6);
all_test2_sparse_percent_correct_updown = zeros(num_its,6);
RF_shape = 1;
for num_RFs = [48]
for hiddenunits = [10]
for proportion = .05 : .1 : .95

RF_vert_size = RF_horiz_size*RF_shape;

for iteration = 1:num_its

    %generate RFs
    % x location | y location | tuning
    RFs = [randi(stim_size,num_RFs,2)-center_location [ones(num_RFs/2,1); 2*ones(num_RFs/2,1)]];
    %RFs(:,1) = corticalSample(size(RFs,1));
    rng(cputime);
    %RFs = corticalSample3(num_RFs);
    %RFs = [RFs [ones(num_RFs/2,1); 2*ones(num_RFs/2,1)]]; %changed this for only 2 RFs
    RF_images = zeros(stim_size,stim_size,num_RFs);
    [X,Y] = meshgrid(-(center_location-1):(center_location-1),-(center_location-1):(center_location-1));

    for RF_num = 1:num_RFs
        RF_ecc = sqrt((RFs(RF_num,1)+51)^2 + RFs(RF_num,2)^2);
        horiz_size = (RF_horiz_size/2)*(1+RF_scaling_slope*(RF_ecc-35.3554)/35.3554);
        vert_size = (RF_vert_size/2)*(1+RF_scaling_slope*(RF_ecc-35.3554)/35.3554);
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
            stimuli(:,:,ceil(4*stim_num/num_all_stims)); %place target at center
        
            %flanker #1
        flanker_location = flanker_positions(randi(length(flanker_positions)));
        if flanker_location > center_location - 20 && flanker_location < center_location + 20
            flanker_location(2) = limited_flanker_positions(randi(length(limited_flanker_positions)));
        else
            flanker_location(2) = flanker_positions(randi(length(flanker_positions)));
        end

        flanker_location = flanker_location(randperm(2)); %randomize (x,y) order 

        flanker_type = randi(4);
        training_flanker_data(stim_num,1:3) = [flanker_location flanker_type];

        stim_image(flanker_location(1)-9:flanker_location(1)+10,flanker_location(2)-9:flanker_location(2)+10) = ...
            stim_image(flanker_location(1)-9:flanker_location(1)+10,flanker_location(2)-9:flanker_location(2)+10) + ...
            stimuli(:,:,flanker_type);

        %flanker #2
        flanker_location = flanker_positions(randi(length(flanker_positions)));
        if flanker_location > center_location - 20 && flanker_location < center_location + 20
            flanker_location(2) = limited_flanker_positions(randi(length(limited_flanker_positions)));
        else
            flanker_location(2) = flanker_positions(randi(length(flanker_positions)));
        end

        flanker_location = flanker_location(randperm(2)); %randomize (x,y) order 

        flanker_type = randi(4);
        training_flanker_data(stim_num,4:6) = [flanker_location flanker_type];

        stim_image(flanker_location(1)-9:flanker_location(1)+10,flanker_location(2)-9:flanker_location(2)+10) = ...
            stim_image(flanker_location(1)-9:flanker_location(1)+10,flanker_location(2)-9:flanker_location(2)+10) + ...
            stimuli(:,:,flanker_type);
        stim_image(stim_image > 1) = 1;

        training_stims(:,:,stim_num) = stim_image;
    end

    %test set 2 horizontal vs. vertical flankers definitely needs to be
    %changed to incorporate type 4 flanker
%     test2list = [1 1 1 1 2 2 2 2 3 3 3 3 1 1 1 1 2 2 2 2 3 3 3 3;
%                  2 3 2 3 1 3 1 3 1 2 1 2 0 0 0 0 0 0 0 0 0 0 0 0;
%                  2 3 3 2 1 3 3 1 1 2 2 1 0 0 0 0 0 0 0 0 0 0 0 0;
%                  0 0 0 0 0 0 0 0 0 0 0 0 2 3 2 3 1 3 1 3 1 2 1 2;
%                  0 0 0 0 0 0 0 0 0 0 0 0 2 3 3 2 1 3 3 1 1 2 2 1];
%                  
%     test_stims2 = zeros(stim_size,stim_size,size(test2list,2));
%              
%     for stim_num = 1:size(test2list,2)
%          
%         stim_image = zeros(stim_size,stim_size);
%         
%         stim_image(center_location-9:center_location+10,center_location-9:center_location+10) = ...
%             stim_image(center_location-9:center_location+10,center_location-9:center_location+10) + ...
%             stimuli(:,:,test2list(1,stim_num)); %place target at center
% 
%         %left flanker
%         if test2list(2,stim_num)
%         stim_image(center_location-9:center_location+10,center_location-34:center_location-15) = ...
%             stim_image(center_location-9:center_location+10,center_location-34:center_location-15) + ...
%             stimuli(:,:,test2list(2,stim_num));end
%     
%         %right flanker
%         if test2list(3,stim_num)
%         stim_image(center_location-9:center_location+10,center_location+16:center_location+35) = ...
%             stim_image(center_location-9:center_location+10,center_location+16:center_location+35) + ...
%             stimuli(:,:,test2list(3,stim_num));end
% 
%         %upper flanker
%         if test2list(4,stim_num)
%         stim_image(center_location-34:center_location-15,center_location-9:center_location+10) = ...
%             stim_image(center_location-34:center_location-15,center_location-9:center_location+10) + ...
%             stimuli(:,:,test2list(4,stim_num));end
%     
%         %lower flanker
%         if test2list(5,stim_num)
%         stim_image(center_location+16:center_location+35,center_location-9:center_location+10) = ...
%             stim_image(center_location+16:center_location+35,center_location-9:center_location+10) + ...
%             stimuli(:,:,test2list(5,stim_num));end
%     
%         test_stims2(:,:,stim_num) = stim_image;
%     
%     end
    
%     training_stims = cat(3,training_stims,test_stims2); %add test_stims2 for computation of RF responses
    
    
    %generate vector of RF outputs
    RF_responses = zeros(num_RFs,size(training_stims,3)); 
    for stim_num = 1:size(training_stims,3)
        current_stim = training_stims(:,:,stim_num);
        filtered_stim = cat(3,normxcorr2e(filters(:,:,1),current_stim,'same'),normxcorr2e(filters(:,:,2),current_stim,'same'));

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

    training_stims = training_stims(:,:,1:num_all_stims); %remove test_stims2
    test_RF_responses2 = RF_responses(:,num_all_stims+1:end);
    RF_responses(:,num_all_stims+1:end) = [];
    
    %keyboard
    %divide training and test data
    test_indices = [(1:num_test_stims/4) (num_all_stims/4 + 1:num_all_stims/4 + num_test_stims/4) (2*num_all_stims/4 + 1:2*num_all_stims/4 + num_test_stims/4) (3*num_all_stims/4 + 1:3*num_all_stims/4 + num_test_stims/4)];
    test_stims = training_stims(:,:,test_indices);
    %keyboard
    training_stims(:,:,test_indices) = []; %remove test stimuli from training data

    training_RF_responses = RF_responses;
    test_RF_responses = training_RF_responses(:,test_indices);
    training_RF_responses(:,test_indices) = []; %remove test stimuli from training data


    training_targets = [repmat([1 0 0 0]',[1 num_training_stims/4]) repmat([0 1 0 0]',[1 num_training_stims/4]) repmat([0 0 1 0]',[1 num_training_stims/4])  repmat([0 0 0 1]',[1 num_training_stims/4])];
    test_targets = [repmat([1 0 0 0]',[1 num_test_stims/4]) repmat([0 1 0 0]',[1 num_test_stims/4]) repmat([0 0 1 0]',[1 num_test_stims/4]) repmat([0 0 0 1]',[1 num_test_stims/4])];
    test2_targets = repmat([repmat([1 0 0]',[1 4]) repmat([0 1 0]',[1 4]) repmat([0 0 1]',[1 4])],[1 2]); %this is still broken
    
    %train network
    net = patternnet(hiddenunits);
    net.trainParam.showWindow = false;
    net.trainParam.showCommandLine = false;
    net = train(net,training_RF_responses,training_targets);

    %test network
    predictions = net(test_RF_responses);
    prediction_classes = vec2ind(predictions);
%    disp(prediction_classes)
    percent_correct = mean(prediction_classes == vec2ind(test_targets));    
    all_percent_correct(iteration) = percent_correct;
    rng(cputime);
    keyboard
    %take out some inputs
%    input_weights = net.IW;
%    proportion_drop = .6;
%    input_weights{1}(:,randsample(1:size(input_weights{1},2),ceil(proportion_drop*size(input_weights{1},2)))) = 0;
%    net.IW = input_weights;
%     
    %retrain on fewer RFs
    drop_indices = randsample(1:num_RFs,ceil(num_RFs*proportion));
    training_RF_responses(drop_indices,:) = [];
    test_RF_responses(drop_indices,:) = [];
    test_RF_responses2(drop_indices,:) = [];
    net = patternnet(hiddenunits);
    net.trainParam.showWindow = false;
    net.trainParam.showCommandLine = false;
    net = train(net,training_RF_responses,training_targets);
    
    %original test set
    predictions = net(test_RF_responses);
    prediction_classes = vec2ind(predictions);
    percent_correct = mean(prediction_classes == vec2ind(test_targets));
    all_sparse_percent_correct(iteration,round((1-RF_shape)*10+1)) = percent_correct;
    
    %test network on test2
%     predictions = net(test_RF_responses2);
%     prediction_classes = vec2ind(predictions);
%     correct_trials = prediction_classes == vec2ind(test2_targets);
% 
%     percent_correct_leftright = mean(correct_trials(1:12)); %this eventually needs to be changed
%     percent_correct_updown = mean(correct_trials(13:24)); %and this
%     
%     all_test2_sparse_percent_correct_leftright(iteration,round((1-RF_shape)*10+1)) = percent_correct_leftright;
%     all_test2_sparse_percent_correct_updown(iteration,round((1-RF_shape)*10+1)) = percent_correct_updown;
%     
    disp(RF_shape)
    disp(iteration)
    
    %save(['/Users/jasonfischer/Dropbox/MATLAB/crowding_model/' save_name '.mat'],'all_percent_correct','all_sparse_percent_correct',...
    %    'all_test2_sparse_percent_correct_leftright','all_test2_sparse_percent_correct_updown');

    
end %end iterations

namepart = num2str(proportion);
namepart = namepart(3:end);
save(['C:/Users/Anna/Desktop/Crowding Model/Tumbling Ts/' 'RFsize_' namepart '_' 'NumRFs_' num2str(num_RFs) 'hiddenunits' num2str(hiddenunits) '.mat'],'all_percent_correct','all_sparse_percent_correct');
%        'all_test2_sparse_percent_correct_leftright','all_test2_sparse_percent_correct_updown');
end
end
end %end RF shape loop


% end %end main function