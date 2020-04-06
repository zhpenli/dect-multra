%-------------------------------------------------------------------------
%
%   Run this file to obtain decompositions of DECT-MULTRA method.
%   In this file, we set different weights for common-material (r=1) and
%   cross-material (r=2) models. Here, beta_1 and beta_2 indicate the
%   weights for common and cross material models, respectively. 
%
%   Considering different materials may have different sparsities and different
%   density values, in this file we set different gamma values for water 
%   (gamma_1) and bone (gamma_3). These beta and gamma values need to be 
%   carefully tuned to achieve the best image decomposition quality.
%   
% 
%   Zhipeng Li, UM-SJTU Joint Institute, Shanghai Jiao Tong University
%   
%-------------------------------------------------------------------------

clear all; close all; clc

for beta1 = [50]
    for beta2 = [50]
        for gamma1 = [0.13]
            for gamma2 = [0.09]
                for gamma3 = [0.13]
                    clc
                    addpath(genpath('../../toolbox'));
                    %% setup target geometry and weight
                    down = 1; % downsample rate
                    ig = image_geom('nx',512,'fov',50,'down',down);
                    ig.mask = ig.circ > 0;
                    %% load external parameter
                    I0 = 1e6; % photon intensity
                    dir = ['../../tmp_ele/dose1e6'];
                    % change intial image when I0 is changed!
                    printm('Loading xfbp, y, transforms...');
                    load([dir '/x.mat']); % direct inversion
                    load([dir '/y.mat']);
                    load([dir '/A.mat']);
                    load([dir '/wi.mat']);
                    load([dir '/1e6EP_l2b8_del0.01_l2b8.5_del0.02_iter500.mat']); % EP 
                    xw = xfis(:,:,1); xb = xfis(:,:,2);
                    x_msk = cat(1, xw(ig.mask), xb(ig.mask)); clear xwtmp xbtmp;

                    wi = cat(1, wi(1,:)', wi(2,:)');

                    yhtmp = y(:,:,1); yltmp = y(:,:,2);
                    y_msk = cat(1, yhtmp(ig.mask), yltmp(ig.mask));
                    N_p = numel(xw(ig.mask)); 

                    %load transform
                    load('../Train/Unitary_5slice_l64_stride1_l031_iter2000_numBlock15_gamma0.21.mat');
                    mOmega1 = info.mOmega; clear info;
                    load('../../DECT_CULTRA/Train/Unitary_slice5_block10_SldDist1_iter2000_gamma0.17.mat');
                    mOmega2 = info.mOmega; clear info;
                    mOmega3 = mOmega1; 

                    %load true image
                    load([dir '/xtrue.mat']);
                    xtrue = single(xtrue);

                    %% set up parameters and regularizer
                    ImgSiz1 = [ig.nx ig.ny 2];   ImgSiz2 = [ig.nx, ig.ny]; %  image size
                    PatSiz1 = [8 8 2];   PatSiz2 = [8 8];   %  patch size
                    SldDist1 = [1 1 1];  SldDist2 = [1 1];  %  sliding distance

                    beta = [beta1 beta2];
                    gamma = [gamma1 gamma2 gamma3];

                    iter = 1; % I--inner Iteration
                    nouter = 500; %K--outer Iteration
                    CluInt = 1;            % Clustering Interval

                    printm('Pre-calculating majorization matrix M...');
                    % pre-compute D_R
                    numBlock = [size(mOmega1, 3), size(mOmega2, 3), size(mOmega3, 3)];

                    PP = im2colstep(ones(ImgSiz1,'single'), PatSiz1, SldDist1);
                    PatNum = size(PP, 2);
                    clear maxLambda1 maxLambda2 maxLambda3;

                    R = unitary_Reg_MULTRA(ig.mask, ImgSiz1, ImgSiz2, PatSiz1, SldDist1, PatSiz2, SldDist2,...
                        beta, gamma, mOmega1, mOmega2, mOmega3, numBlock, CluInt); 
                    clear xwtmp xbtmp yhtmp yltmp

                    %% Algorithm Starts 
                    SqrtPixNum = sqrt(sum(ig.mask(:)>0));
                    info = struct('intensity',I0,'ImgSiz',ImgSiz1,'SldDist',SldDist1,'beta',beta,'gamma',gamma,...
                        'omg1',mOmega1,'omg2',mOmega2,'omg3',mOmega3,'nIter',iter,'CluInt',CluInt,...
                        'vIdx1',[],'vIdx2',[],'vIdx3',[],'ClusterMap',[], 'RMSE',[],'SSIM',[],'relE',[],'perc',...
                         [],'idx_change_perc',[],'cost',[]);
                     
                    fprintf('Decompose starts...\n');

                    idx_old = single(ones(1,PatNum));
                    for ii = 1:nouter
 
                        ticker(mfilename, ii, nouter)
                        info.RMSE(1,ii) = norm(col(xw - xtrue(:,:,1))) / SqrtPixNum;
                        info.RMSE(2,ii) = norm(col(xb - xtrue(:,:,2))) / SqrtPixNum;
                        fprintf('WRMSE = %g, BRMSE = %g, ', info.RMSE(1,ii), info.RMSE(2,ii));  

                        info.SSIM(1,ii)= ssim(xw, xtrue(:,:,1));
                        info.SSIM(2,ii)= ssim(xb, xtrue(:,:,2));
                        fprintf('WSSIM = %g, BSSIM = %g, ', info.SSIM(1,ii), info.SSIM(2,ii)); 

                        [x_msk, cost] = unitary_pwls(x_msk,  A, y_msk, wi, R, 0, iter);

                        [info.perc(:,ii),info.vIdx1,info.vIdx2,info.vIdx3,info.IDX_water,...
                            info.IDX_cross, info.IDX_bone] = R.nextOuterIter();
                        fprintf('perc = %g, perc = %g, perc = %g\n', info.perc(1,ii),info.perc(2,ii),info.perc(3,ii));
                        fprintf('Patch number: %d, %d, %d \n', size(info.IDX_water,2), size(info.IDX_cross,2), size(info.IDX_bone,2))

                        info.cost(:,ii) = cost;
                        xw = embed(x_msk(1:1:end/2,1),ig.mask); 
                        xb = embed(x_msk(end/2+1:1:end,1),ig.mask); 
                       % xw = max(xw, 0);  xb = max(xb, 0);
                        figure(110), imshow(xw, [0.6 1.4]); drawnow;
                        figure(111), imshow(xb, [0 0.8]); drawnow;
                    end


                    close all; 
                    xw = embed(x_msk(1:1:end/2,1),ig.mask); 
                    xb = embed(x_msk(end/2+1:1:end,1),ig.mask); 
                    xfista = cat(3,xw,xb);
                    info.x = xfista; 
                    rmim = cat(1, info.RMSE(1,:),info.SSIM(1,:),info.RMSE(2,:),info.SSIM(2,:));

                   save(sprintf('./weight_learn_block15_10_gam0.21_0.17_beta%2.2g_%2.2g_gamma%2.3g_%2.3g_%2.3g_iter%d_outer%d.mat',... 
                       beta(1), beta(2), gamma(1,1), gamma(1,2),gamma(1,3),iter, nouter),'info');
                    clearvars -except beta1 gamma1 gamma2 gamma3 PatNum
                end
            end
        end
    end
end

figure name 'water RMSE'
plot(info.RMSE(1,:))
xlabel('Number Iteration','fontsize',18)
ylabel('RMSE','fontsize',18)
figure name 'bone RMSE'
plot(info.RMSE(2,:))
xlabel('Number Iteration','fontsize',18)
ylabel('RMSE','fontsize',18)


% plot(info.cost(1,:),'linewidth',2)
% grid on
% xlabel('Number of Iterations','fontsize',18)
% ylabel('Objective Function','fontsize',18)

