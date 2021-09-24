Nirx = dir('C:\Users\Bruno\Documents\Experimento\nirs\dados_bruno\nirx\Resting\resting_oxy\correlacoes\*');
a = dlmread(Nirx(9).name);

for j=10:26
    b = dlmread(Nirx(j).name);
    a(:,:,(j-8)) = b;
    clear b;
end

b = dlmread('comportamental4.txt');

%----------------- INPUTS -----------------

all_mats = a;
all_behav = b;

%threshold for feature selection
thresh = 0.05;
%------------------------------

no_sub = size(all_mats,3);
no_node = size(all_mats,1);

behav_pred_pos = zeros(no_sub,1);
behav_pred_neg = zeros(no_sub,1);

for leftout = 1:no_sub;
    fprintf('\n Leaving out subj # %6.3f',leftout);
    
    %leave out subject from matrices and behavior
    
    train_mats = all_mats;
    train_mats(:,:,leftout) = [];
    train_vcts = reshape(train_mats,[],size(train_mats,3));
    
    train_behav = all_behav;
    train_behav(leftout) = [];
    
    %correlate all edges with behavior
    
    [r_mat,p_mat] = corr(train_vcts',train_behav);
    
    r_mat = reshape(r_mat,no_node,no_node);
    p_mat = reshape(p_mat,no_node,no_node);
    
    % set threshold and define masks
    
    pos_mask = zeros(no_node,no_node);
    neg_mask = zeros(no_node,no_node);
    
    pos_edges = find(r_mat > 0 & p_mat < thresh);
    neg_edges = find(r_mat < 0 & p_mat < thresh);
    
    pos_mask(pos_edges) = 1;
    neg_mask(neg_edges) = 1;
    
    %get sum of all edges in TRAIN subs (divide by 2 to control for the
    %fact that matrices are symmetric)
    
    train_sumpos = zeros(no_sub-1,1);
    train_sumneg = zeros(no_sub-1,1);
    
    for ss = 1:size(train_sumpos);
        train_sumpos(ss) = sum(sum(train_mats(:,:,ss).*pos_mask))/2;
        train_sumneg(ss) = sum(sum(train_mats(:,:,ss).*neg_mask))/2;
    end
    
    %build model on TRAIN subs
    
    fit_pos = polyfit(train_sumpos, train_behav,1);
    fit_neg = polyfit(train_sumneg, train_behav,1);
    
    %Run model on TEST sub
    
    test_mat = all_mats(:,:,leftout);
    test_sumpos = sum(sum(test_mat.*pos_mask))/2;
    test_sumneg = sum(sum(test_mat.*neg_mask))/2;
    
    behav_pred_pos(leftout) = fit_pos(1)*test_sumpos + fit_pos(2);
    behav_pred_neg(leftout) = fit_neg(1)*test_sumneg + fit_neg(2);

%       % build model on TRAIN subs
%       % combining both positive and negative features
%       b = regress(train_behav, [train_sumpos, train_sumneg, ones(no_sub-1,1)]);
%       
%       %run model on TEST sub
%       
%       test_mat = all_mats(:,:,leftout);
%       test_sumpos = sum(sum(test_mat.*pos_mask))/2;
%       test_sumneg = sum(sum(test_mat.*neg_mask))/2;
%       
%       behav_pred(leftout) = b(1)*test_sumpos + b(2)*test_sumneg + b(3);
    
end

% compare predicted and observed scores

[R_pos, P_pos] = corr(all_behav, behav_pred_pos)
[R_neg, P_neg] = corr(all_behav, behav_pred_neg)

figure(1); plot(behav_pred_pos, all_behav,'r.'); lsline
figure(2); plot(behav_pred_neg, all_behav,'b.'); lsline

% %calculate the true prediction correlation
% [true_prediction_r_pos, true_prediction_r_neg] = predict_behavior(all_mats, all_behav);
% 
% %number of iterations for permutation testing
% no_iterations = 1000;
% prediction_r = zeros(no_iterations,2);
% prediction_r(1,1) = true_prediction_r_pos;
% prediction_r(1,2) = true_prediction_r_neg;
% 
% %create estimate distribution of the test statistic
% %via random shuffles of data labels
% for it=2:no_iterations
%     fprintf('\n Performing iteratrion %d out of %d', it, no_iterations);
%     new_behav = all_behav(randperm(no_sub));
%     [prediction_r(it,1), prediction_r(it,2)] = predict_behavior(all_mats, new_behav);
% end
% 
% sorted_prediction_r_pos = sort(prediction_r(:,1),'descend');
% position_pos = find(sorted_prediction_r_pos==true_prediction_r_pos);
% pval_pos = position_pos(1)/no_iterations;
% 
% sorted_prediction_r_neg = sort(prediction_r(:,2),'descend');
% position_neg = find(sorted_prediction_r_neg==true_prediction_r_neg);
% pval_neg = position_neg(1)/no_iterations;


