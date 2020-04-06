%--------------------------------------------------------------------------
%   Run this file to learn the union of common-material transforms.
%   Basis material images should be prepared in advance.
%   Key parameters are the number of clustering classes (denoted by num),
%   and gamma(denoted by gam).
%
%   This file corresponding to the common-material union of ST training 
%   process for DECT-MULTRA method.
%
%   Zhipeng Li, UM-SJTU Joint Institute, Shanghai Jiao Tong University
%
%--------------------------------------------------------------------------
clear ; close all; clc
addpath(genpath('../../toolbox'));
%%%%%%%%%%%%%%%%%%%%%%%%%%% load training data %%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for gam = [0.21]
    for num = [15]
        PatSiz = 8 * [1 1];  % patch size
        SldDist = 1 * [1 1];  % sliding distance
        ImgSiz = [512 512];
        iter = 2000;  % iteration
        gamma = gam;
        numBlock = num;   % cluster number

        patch=[];
        slice = [65 70 75 80 90]; %choose some slices from 3D data to do training
        for ii = 1:length(slice)
             for j = 1:2
                if j == 1
                    tmp = phantom(:,:,slice(ii)).*(phantom(:,:,slice(ii)) > 0.9 & phantom(:,:,slice(ii)) < 1.8);
                else
                    tmp = phantom(:,:,slice(ii)).*(phantom(:,:,slice(ii)) > 1.8 & phantom(:,:,slice(ii)) < 2);
                end
                image(:,:,(ii-1)*2+j) = downsample2(tmp, 2); 
                patch_tmp = im2colstep(single( image(:,:, (ii-1)*2+j )), PatSiz, SldDist); 
                patch_tmp = single(patch_tmp);
                patch = [patch patch_tmp];
             end
        end

        patch = double(patch); 
        numPatch = size(patch, 2);
        fprintf('Length of training set: %d\n', numPatch);

        IDX = randi(numBlock,numPatch,1);

        TransWidth = prod(PatSiz);

        D = kron(dctmtx(PatSiz(1)),dctmtx(PatSiz(2))); % DCT Initialization
        mOmega = zeros(TransWidth,TransWidth,numBlock, 'double');
        for i = 1:numBlock
          mOmega(:,:,i) = D;
        end

        perc = zeros(iter,numBlock,'single'); % sparsity(percentage)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Training %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        for j = 1 : iter
          fprintf('iteration = %d:\n', j);
          for k = 1 : numBlock
            patch_k = patch(:, IDX == k); 
            sparseCode = mOmega(:,:,k) * patch_k;
            sparseCode = sparseCode .*(abs(sparseCode) > gamma);
            perc(j,k) = nnz(sparseCode)/ numel(sparseCode)* 100;
            % transform update
            if (size(patch_k,2) > 0) 
              [Q1,Si,R]=svd(patch_k*sparseCode');
              mOmega(:,:,k)=R*Q1';
            else
              fprintf('patch %g is empty\n', k);
            end
          end
          clear patch_k sparseCode LL2 Q1 sig R B
          fprintf('Cond Number(%d) = %g\n', numBlock, cond(mOmega(:,:,numBlock)));
          fprintf('sparsity(%d) = %g\n', numBlock, perc(j,numBlock));
          error = zeros(numBlock, numPatch, 'double');
          %%%%%%%%% clustering measure %%%%%%%
          for k = 1 : numBlock
            a1 = mOmega(:,:,k) * patch;
            a0 = a1 .*(abs(a1) > gamma);
            error(k, :) = sum((a1-a0).^2,'double') + gamma^2 * sum(abs(a0) > 0);
          end
          %%%%%%%%% clustering %%%%%%%%%%%%%%
          [~, IDX] = min(error, [] ,1);
          show(mOmega);
          clear  error a1 a0
        end
        %%%%%%%%%%%%%%%%%%%%%% check convergency %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        figure();
        for k = 1 : numBlock
          handles(k) = plot(perc(:,k));hold on;
          lables{k} = sprintf('cluster %d',k);
        end
        legend(handles,lables{:});
        xlabel('Number of Iteration','fontsize',18)
        ylabel('Sparity ( % )','fontsize',18)
        %%%%%%%%%%%%%%%%%%%%%% check condition number %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        condTransform = zeros(numBlock, 1);
        for k = 1 : numBlock
          condTransform(k, 1) = cond(mOmega(:, :, k));
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        info = struct('ImgSiz',size(image),'SldDist',SldDist,'gamma',gamma,...
         'numBlock',numBlock,'iter',iter,'mOmega',mOmega,'IDX',IDX ,'perc',perc);
        save(sprintf('Unitary_5slice_l64_stride1_l031_iter%d_numBlock%d_gamma%2.2g.mat',... 
                 iter, numBlock, gamma),'info');
            clearvars -except gam num phantom
            close all; clc
    end
end

function show(mOmega)
    transform = mOmega;
    Taa=[];
    Taaa=[];
    for k = 1:size(transform, 3)
      for i=1:size(transform,1)
        transform(i,:,k) = transform(i,:,k)-min(transform(i,:,k));
        if(max(transform(i,:,k))>0)
          transform(i,:,k)=transform(i,:,k)/(max(transform(i,:,k)));
        end
      end  

      jy=2;cc=1;
      Ta=(max(max(transform(:, :, k))))*ones((8+jy)*7 + 8,(8+jy)*7 + 8);
      for i=1:8+jy:(7*(8+jy))+1
        for j=1:8+jy:(7*(8+jy))+1
          Ta(i:i+7,j:j+7)=reshape((transform(cc,:,k))',8,8);
          cc=cc+1;
        end
      end
      blank = zeros(size(Ta,1),2);
      Taa = cat(2 , Ta , blank);
      Taaa = cat(2 , Taaa , Taa);
    end
    figure(110);imagesc(Taaa);colormap('Gray');axis off;axis image; drawnow
end