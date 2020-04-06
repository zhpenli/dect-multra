%------------------------------------------------------------------------
%
%   This is the main file for the testing process of DECT-CULTRA method.
%
%   There are two parameters (beta and gamma) need to be tuned for the
%   cross-material.
%   
%   Zhipeng Li, UM-SJTU Joint Institute, Shanghai Jiao Tong Univresity
%
%------------------------------------------------------------------------
clear all; close all; clc
%addpath(genpath('/home/zhipeng/Desktop/DECT_ST_ncat'));
for beta_set = [70]
    for gamma_set = [0.07]
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
        load('../Train/Unitary_slice5_block10_SldDist1_iter2000_gamma0.17.mat');
        %load('/home/zhipeng/Desktop/Cross_XCAT/ULTRA/slice5_block15_SldDist1_iter2000_gamma012.mat');
        mOmega = info.mOmega; clear info;

        %load true image
        load([dir '/xtrue.mat']);
        xtrue = single(xtrue);

        %% set up regularizer

        ImgSiz = [ig.nx ig.ny 2]; % image size
        PatSiz = [8 8 2];   % patch size
        SldDist = [1 1 1];  % sliding distance

        beta = beta_set;
        gamma = gamma_set;

        iter = 1; % I--inner Iteration, set to one because of closed-form solution
        nouter = 500; %K--outer Iteration
        CluInt = 1;            % Clustering Interval
        isCluMap = 0;          % The flag of calculating cluster mapping

        printm('Pre-calculating majorization matrix M...');
        % pre-compute D_R
        numBlock = size(mOmega, 3);
        vLambda = [];
        % for k = 1:numBlock
        %     vLambda = cat(1, vLambda, eig(mOmega(:,:,k)' * mOmega(:,:,k)));
        % end
        % maxLambda = max(vLambda); clear vLambda;
        Mtmp = 2 * beta * prod(PatSiz) / prod(SldDist)/2;
        M = cat(1, Mtmp * ones(N_p, 1), Mtmp * ones(N_p,1));

        PP = im2colstep(ones(ImgSiz,'single'), PatSiz, SldDist);
        PatNum = size(PP, 2);
        clear maxLambda1 maxLambda2;

        R = unitary_Reg_CULTRA(ig.mask, ImgSiz, PatSiz, SldDist, beta, gamma, mOmega, numBlock, CluInt); 
        clear xwtmp xbtmp yhtmp yltmp

        %% Algorithm Starts 
        SqrtPixNum = sqrt(sum(ig.mask(:)>0));
        info = struct('intensity',I0,'ImgSiz',ImgSiz,'SldDist',SldDist,'beta',beta,'gamma',gamma,...
            'nIter',iter,'CluInt',CluInt,'transform',mOmega,...
            'vIdx',[],'ClusterMap',[], 'RMSE',[],'SSIM',[],'relE',[],'perc',[],'idx_change_perc',[],'cost',[]);
        fprintf('Decompose starts...\n');

        idx_old = single(ones(1,PatNum));

        for ii = 1:nouter
            %fprintf('Iteration = %d of %d:\n', ii, outiter);
            ticker(mfilename, ii, nouter)
            info.RMSE(1,ii) = norm(col(xw - xtrue(:,:,1))) / SqrtPixNum;
            info.RMSE(2,ii) = norm(col(xb - xtrue(:,:,2))) / SqrtPixNum;
            fprintf('WRMSE = %g, BRMSE = %g, ', info.RMSE(1,ii), info.RMSE(2,ii));  

            info.SSIM(1,ii)= ssim(xw, xtrue(:,:,1));
            info.SSIM(2,ii)= ssim(xb, xtrue(:,:,2));
            fprintf('WSSIM = %g, BSSIM = %g\n', info.SSIM(1,ii), info.SSIM(2,ii)); 

            [x_msk, cost] = unitary_pwls(x_msk,  A, y_msk, wi, R, M,  1, iter);

            [info.perc(:,ii),info.vIdx] = R.nextOuterIter();
            fprintf('perc = %g\n', info.perc(:,ii));

            info.idx_change_perc(:,ii) = nnz(idx_old - info.vIdx)/PatNum;
            fprintf('Idx Change Perc = %g\n', info.idx_change_perc(:,ii));
            idx_old = info.vIdx; 

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
        info.x = xfista; info.omg = mOmega;
        rmim = cat(1, info.RMSE(1,:),info.SSIM(1,:),info.RMSE(2,:),info.SSIM(2,:));
        %info.spa = sparsecode;
        save(sprintf('./unitary_learn_block10_gam0.17_beta%2.2g_gamma%2.2g_iter%d_outer%d.mat', ...
            beta, gamma, iter, nouter),'info');
        clearvars -except beta_set gamma_set
    end
end

%% Calculate clusterMap using ClusterMap.c
% if(isCluMap == 1)
%    info.ClusterMap = ClusterMap(ImgSiz, PatSiz, SldDist, info.vIdx, PatNum, numBlock);
%    transform = info.mOmega;
%     Taa=[];
%     Taaa=[];
%     for k = 1:size(transform, 3)
%       for i=1:size(transform,1)
%         transform(i,:,k) = transform(i,:,k)-min(transform(i,:,k));
%         if(max(transform(i,:,k))>0)
%           transform(i,:,k)=transform(i,:,k)/(max(transform(i,:,k)));
%         end
%       end  
%       kb = 7;  counter = 1; ka = 15;
%       jy = 1; % control the line interval between different patches
% 
%         Ta = zeros((8+jy)*ka+8,(16+jy)*kb+16); % use "ones" for different looking
%         for i=1:8+jy:ka*(8+jy)+1
%             for j=1:16+jy:(kb*(16+jy))+1
%                if(counter<=size(transform,1))    
%                Ta(i:i+7,j:j+15) = reshape(transform(counter,:,k),8,16);
%                else
%                    Ta(i:i+7,j:j+15) = 1;
%                end
%                counter=counter+1;
%             end
%         end
% 
%       blank = zeros(size(Ta,1),2);
%       Taa = cat(2 , Ta , blank);
%       Taaa = cat(2 , Taaa , Taa);
%     end
%     figure(220);imagesc(Taaa);colormap('Gray');axis off;axis image;
%     for j = 1:numBlock
%    % for j = 5
%         figure(j), imshow( (info.ClusterMap(:,:,1) == j) .* info.x(:,:,1), [0.6 1.4]);
%     end
% end
% --------------------- plot RMSE Curve -------------------------------
figure name 'water RMSE'
plot(info.RMSE(1,:))
xlabel('Number Iteration','fontsize',18)
ylabel('RMSE','fontsize',18)
figure name 'bone RMSE'
plot(info.RMSE(2,:))
xlabel('Number Iteration','fontsize',18)
ylabel('RMSE','fontsize',18)
% ------------------------------------------------------------------------

