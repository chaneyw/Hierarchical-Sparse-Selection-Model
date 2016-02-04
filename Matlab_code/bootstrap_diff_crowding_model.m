function [results] = bootstrap_diff_crowding_model(x,y)

num_its = 1000;
results = zeros(num_its,1);

for iteration = 1:num_its
    a = x(randi(length(x),100,1));
    b = y(randi(length(y),100,1));
    
    results(iteration) = mean(a) - mean(b);
    
end



end