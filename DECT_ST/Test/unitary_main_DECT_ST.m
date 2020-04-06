%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   run this file to obtain the decompositions of DECT-ST method.
%
%   For each material, there are two parameters (beta and gamma), which need
%   to be carefully tuned.
% 
%   Zhipeng Li, UM-SJTU Joint Institute, Shanghai Jiao Tong University
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


clear all; close all; clc
addpath(genpath('../../TMI_code/toolbox'));
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

%load transforms
load('../Train/Unitary_water_den_l64_stride1_l031_iter2000_gamma0.12.mat');
mOmega1 = mOmega;
load('../Train/Unitary_bone_den_l64_stride1_l031_iter2000_gamma0.15.mat');
mOmega2 = mOmega;

%load true image
load([dir '/xtrue.mat']);
xtrue = single(xtrue);

%% set up regularizer

ImgSiz = [ig.nx ig.ny]; % image size
PatSiz = 8 * [1 1];   % patch size
SldDist = 1 * [1 1];  % sliding distance

beta1 = 5e1; beta2 = 7e1;

gamma1 = 0.03; gamma2 = 0.04;

iter = 1; % I--inner Iteration
nouter = 1000; %K--outer Iteration

printm('Pre-calculating majorization matrix M...');
M1 = 2 * beta1 * prod(PatSiz)/ prod(SldDist);
M2 = 2 * beta2 * prod(PatSiz)/ prod(SldDist);
M = cat(1, M1 * ones(N_p,1), M2 * ones(N_p,1));
clear maxLambda1 maxLambda2;

R = unitary_Reg_st(ig.mask, PatSiz, ImgSiz, SldDist, beta1, beta2, gamma1,...
    gamma2, mOmega1, mOmega2);
clear xwtmp xbtmp yhtmp yltmp

%% Algorithm Starts 
SqrtPixNum = sqrt(sum(ig.mask(:)>0));
info = struct('cost',[], 'RMSE',[], 'SSIM', []);
fprintf('Decompose starts...\n');

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
    R.nextOuterIter(); 
    
    info.cost(:,ii) = cost;
    xw = embed(x_msk(1:1:end/2,1),ig.mask); 
    xb = embed(x_msk(end/2+1:1:end,1),ig.mask); 
  
    figure(110), imshow(xw, [0.7 1.3]); drawnow;
    figure(111), imshow(xb, [0 0.8]); drawnow;
end
    
close all; 
xw = embed(x_msk(1:1:end/2,1),ig.mask); 
xb = embed(x_msk(end/2+1:1:end,1),ig.mask); 
xfista = cat(3,xw,xb);
info.x = xfista; info.omg1 = mOmega1; info.omg2 = mOmega2;
rmim = cat(1, info.RMSE(1,:),info.SSIM(1,:),info.RMSE(2,:),info.SSIM(2,:));
% save(sprintf('%1.0gST_max_beta%2.2g_beta%2.2g_gamma%2.2g_gamma%2.2g_iter%d_outer%d.mat', ...
%    I0, beta1, beta2, gamma1, gamma2, iter, nouter),'info');
%% show reconstructed material images
% x_true = sum(xtrue, 3);
% xep = sum(xfis ,3);
% x_fis = sum(info.x, 3);
% iy = ig.ny/2+1; ix = 1:ig.nx;
% %iy = 1:ig.ny; ix = ig.nx/2 + 1;
% pro = @(x) x(ix,iy);
% % pseudo-density
% xpse_true = xtrue(:,:,1) + xtrue(:,:,2) * A(2) / A(1);
% plot([ pro(x_true) pro(x_fis) pro(xep) pro(sum(x,3))])
% legend(  'dens true', 'ST', 'EP', 'Direct')

figure name 'water RMSE'
plot(info.RMSE(1,:))
xlabel('Number Iteration','fontsize',18)
ylabel('RMSE','fontsize',18)
figure name 'bone RMSE'
plot(info.RMSE(2,:))
xlabel('Number Iteration','fontsize',18)
ylabel('RMSE','fontsize',18)
% 
figure name 'water compare' 
imshow(cat(2, info.x(:,:,1), xfis(:,:,1)), [0.6 1.4]);colorbar; 
figure name 'bone compare' 
imshow(cat(2, info.x(:,:,2), xfis(:,:,2)), [0 0.8]);colorbar;



