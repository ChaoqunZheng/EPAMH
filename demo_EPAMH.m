clear all;
warning off;
clc;
%% Dataset Loading
load mir_cnn.mat;
fprintf('MIR Flickr_CNN dataset loaded...\n');

%% Training & Evaluation Process
fprintf('\n============================================Start training EPAMH============================================\n');
run = 5;
bit = 8;
for j = 1:run
    
    % PCA X
    XX = I_tr;
    num_training = size(XX,1);
    sampleMeanX = mean(XX,1);
    XX = (XX - repmat(sampleMeanX,size(XX,1),1));
    [pcx, ~] = eigs(cov(XX(1:num_training,:)),bit);
    XX = XX * pcx;
    
    %PCA Y
    YY = T_tr;
    num_training = size(YY,1);
    sampleMeanY = mean(YY,1);
    YY = (YY - repmat(sampleMeanY,size(YY,1),1));
    [pcy,~] = eigs(cov(YY(1:num_training,:)),bit);
    YY = YY * pcy;
    
    %% Offline Training
    [R1,R2,mu1,mu2] = solve_EPAMH(XX, YY, 50);
    
    %% Online Query   
    fprintf('start evaluating for query samples...\n');
    %PCA X
    XX = I_te;
    XX = XX * pcx;
    % %PCA Y
    YY = T_te;
    YY = YY * pcy;
    [B_te] = Query_EPAMH(XX,YY, 50,R1,R2,mu1,mu2);
    
    fprintf('start evaluating for database samples...\n');
    %PCA X
    XX = I_db;
    XX = XX * pcx;
    % %PCA Y
    YY = T_db;
    YY = YY * pcy;
    [B_db] = Query_EPAMH(XX, YY, 50, R1, R2, mu1, mu2);
    
    % Evaluation
    Dhamm = hammingDist(B_db, B_te);
    [MAP] = perf_metric4Label(L_db, L_te,Dhamm);
    map(j) = MAP;
    fprintf('============================================%d bits EPAMH mAP over %d iterations:%.4f=============================================\n', bit, run, map);
end

