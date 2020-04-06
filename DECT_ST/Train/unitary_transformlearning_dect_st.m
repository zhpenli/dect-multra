%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This file is used to learn the sparsifying transforms (ST) for water and bone.
% corresponding to the training process of DECT-ST.
% By setting water = 1, learn ST for water, else, learn ST for bone.
%
% Zhipeng Li, UM-SJTU Joint Institute
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear ; close all; clc
addpath(genpath('../../toolbox'));

%%%%%%%%%%%%%%%%%%%%%%%%  Load Training data  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

pix2cm = 1/0.05;
cm2HU = 1000/0.2;

water = 0;

       l = 64; % patch size
  stride = 1;  % overlapping stride      
lambada0 = 31;  % 0.031 is fine 
    iter = 2000; % iteration
   gamma = 0.15;   

if water
    phantom = phantom.*(phantom > 0.9 & phantom < 1.8);
else
    phantom = phantom.*(phantom > 1.8 & phantom < 2);
end
patch=[];
for ii = [65 70 75 80 90]  % training data
   image = downsample2(phantom(:, :, ii), 2); 
   patch_tmp = im2colstep(single(image), sqrt(l)*[1 1], stride*[1 1]);   
   patch = [patch patch_tmp];
end
 
mOmega = kron(dctmtx(sqrt(l)),dctmtx(sqrt(l)));% DCT Initialization
    
perc=[]; condTransform=[]; % sparsity(percentage) 
for j=1:iter
    fprintf('iteration = %d:\n', j);
    % sparse coding
    sparseCode = mOmega * patch;
    sparseCode = sparseCode.*(abs(sparseCode) > gamma);
    perc(j) = nnz(sparseCode)/ numel(sparseCode) * 100;
    fprintf('sparsity  = %g:\n', perc(j));
    % transform update
    [Q1,Si,R] = svd(patch*sparseCode');
%    sig = diag(Si);
%     gamm = (1/2)*(sig + (sqrt((sig.^2) + 2*lambada)));
%     B = R*(diag(gamm))*Q1';
    mOmega = R*Q1';
  
    show(mOmega);
   
%     condTrans(j) = cond(transform); 
end
save(sprintf('Unitary_water_den_l%d_stride%d_iter%d_gamma%2.2g.mat', l, stride, iter, gamma),'mOmega');     
% save('mOmega.mat','mOmega')
% save('perc.mat','perc')

% check convergency  
figure(); plot(perc);
xlabel('Number of Iteration','fontsize',18)
ylabel('Sparity ( % )','fontsize',18) 

function show(mOmega)
% show learnt transform
for i=1:size(mOmega,1)
    mOmega(i,:)=mOmega(i,:)-min(mOmega(i,:));
    if(max(mOmega(i,:))>0)
      mOmega(i,:)=mOmega(i,:)/(max(mOmega(i,:)));
    end
end
   
jy=1;cc=1;
Ta=(max(max(mOmega)))*ones((8+jy)*7 + 8,(8+jy)*7 + 8);
for i=1:8+jy:(7*(8+jy))+1
    for j=1:8+jy:(7*(8+jy))+1
       Ta(i:i+7,j:j+7)=reshape((mOmega(cc,:))',8,8);
       cc=cc+1;
    end
end
imagesc(Ta);colormap('Gray');axis off;axis image; 
drawnow;
end   
    