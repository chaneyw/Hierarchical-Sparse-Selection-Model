function [] = crowding_model_filters_heatmap()

stim_size = 101;
center_location = ceil(stim_size/2);

RF_horiz_size = 50;
RF_vert_size = 50;
RF_scaling_slope = .75;

num_RFs = 24; %divisible by 3
num_training_stims = 120; %divisible by 3

num_its = 200;

load('/Users/jasonfischer/Dropbox/MATLAB/crowding_model/filters.mat','filters'); %load filters

save_name = input('Name of save file: ','s');

overall_performance_image = zeros(stim_size,stim_size);

for iteration = 1:num_its

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


    % training stimuli
    flanker_positions = (10:stim_size-10);
    limited_flanker_positions = [(10:center_location-20) (center_location+20:stim_size-10)];
    training_flanker_data = zeros(num_training_stims,6);

    training_stims = zeros(stim_size,stim_size,num_training_stims); %this will include test stims, then they're pulled out
    for stim_num = 1:num_training_stims
        stim_image = zeros(stim_size,stim_size);
        stim_image(center_location-9:center_location+10,center_location-9:center_location+10) = ...
            stim_image(center_location-9:center_location+10,center_location-9:center_location+10) + ...
            filters(:,:,ceil(3*stim_num/num_training_stims)); %place target at center

        for flanker_num = 1:2 %two flankers
        
            flanker_location = flanker_positions(randi(length(flanker_positions)));
            if flanker_location > center_location - 20 && flanker_location < center_location + 20
                flanker_location(2) = limited_flanker_positions(randi(length(limited_flanker_positions)));
            else
                flanker_location(2) = flanker_positions(randi(length(flanker_positions)));
            end

            flanker_location = flanker_location(randperm(2)); %randomize (x,y) order 

            flanker_type = randi(3);
            training_flanker_data(stim_num,(3*flanker_type)-2:3*flanker_type) = [flanker_location flanker_type];

            stim_image(flanker_location(1)-9:flanker_location(1)+10,flanker_location(2)-9:flanker_location(2)+10) = ...
                stim_image(flanker_location(1)-9:flanker_location(1)+10,flanker_location(2)-9:flanker_location(2)+10) + ...
                filters(:,:,flanker_type); %place flanker

        end
        
        stim_image(stim_image > 1) = 1;
        training_stims(:,:,stim_num) = stim_image;
    end
    
    
    %generate vector of RF outputs
    RF_responses = zeros(num_RFs,num_training_stims);
    for stim_num = 1:num_training_stims
        current_stim = training_stims(:,:,stim_num);
        filtered_stim = cat(3,normxcorr2e(filters(:,:,1),current_stim,'same'),normxcorr2e(filters(:,:,2),current_stim,'same'),normxcorr2e(filters(:,:,3),current_stim,'same'));

        for RF_num = 1:num_RFs

            RF_pooled = RF_images(:,:,RF_num).*filtered_stim(:,:,RFs(RF_num,3)); %pool over RF using correct filter
            RF_response = max(max(RF_pooled));

            if RF_response < 0     %graded response
                RF_response = 0;
            end

            RF_responses(RF_num,stim_num) = RF_response;
        end
    end

    training_targets = [repmat([1 0 0]',[1 num_training_stims/3]) repmat([0 1 0]',[1 num_training_stims/3]) repmat([0 0 1]',[1 num_training_stims/3])];

    %train network
    net = patternnet(10);
    net.trainParam.showWindow = false;
    net.trainParam.showCommandLine = false;
    net = train(net,RF_responses,training_targets);
    
    %take a sparse sample of RFs
    input_weights = net.IW;
    proportion_drop = .6;
    input_weights{1}(:,randsample(1:size(input_weights{1},2),ceil(proportion_drop*size(input_weights{1},2)))) = 0;
    net.IW = input_weights;
    
    
    %test over a grid of image locations  
    test_list = [1 1 2 2 3 3;
                 2 3 1 3 1 2];
    
    test_targets = [repmat([1 0 0]',[1 2]) repmat([0 1 0]',[1 2]) repmat([0 0 1]',[1 2])];
        
    performance_image = zeros(stim_size,stim_size);
    for x_location = 10:1:stim_size-10
        for y_location = 10:1:stim_size-10
            
            %generate test stimuli w/ flanker at x,y location
            test_stims = zeros(stim_size,stim_size,size(test_list,2));
            
            for stim_num = 1:size(test_list,2)

                stim_image = zeros(stim_size,stim_size);

                %place target at center
                stim_image(center_location-9:center_location+10,center_location-9:center_location+10) = ...
                    stim_image(center_location-9:center_location+10,center_location-9:center_location+10) + ...
                    filters(:,:,test_list(1,stim_num));

                %place flanker
                stim_image(y_location-9:y_location+10,x_location-9:x_location+10) = ...
                    stim_image(y_location-9:y_location+10,x_location-9:x_location+10) + ...
                    filters(:,:,test_list(2,stim_num));

                stim_image(stim_image > 1) = 1;
                test_stims(:,:,stim_num) = stim_image;

            end
            
            %RF responses for test stimuli
            RF_responses = zeros(num_RFs,size(test_list,2));
            for stim_num = 1:size(test_list,2)
                current_stim = test_stims(:,:,stim_num);
                filtered_stim = cat(3,normxcorr2e(filters(:,:,1),current_stim,'same'),normxcorr2e(filters(:,:,2),current_stim,'same'),normxcorr2e(filters(:,:,3),current_stim,'same'));

                for RF_num = 1:num_RFs

                    RF_pooled = RF_images(:,:,RF_num).*filtered_stim(:,:,RFs(RF_num,3)); %pool over RF using correct filter
                    RF_response = max(max(RF_pooled));

                    if RF_response < 0     %graded response
                        RF_response = 0;
                    end

                    RF_responses(RF_num,stim_num) = RF_response;
                end
            end

            %test the model
            predictions = net(RF_responses);
            prediction_classes = vec2ind(predictions);
            percent_correct = mean(prediction_classes == vec2ind(test_targets));
            performance_image(y_location,x_location) = percent_correct;
            
%             disp(x_location)
%             disp(y_location)
            
        end %end y location
    end %end x location
    
    overall_performance_image = overall_performance_image + performance_image;
    save(['/Users/jasonfischer/Dropbox/MATLAB/crowding_model/' save_name '.mat'],'overall_performance_image')
    
    disp(iteration)
    
end %end iterations

overall_performance_image = overall_performance_image/num_its;
save(['/Users/jasonfischer/Dropbox/MATLAB/crowding_model/' save_name '.mat'],'overall_performance_image')

end %end main function