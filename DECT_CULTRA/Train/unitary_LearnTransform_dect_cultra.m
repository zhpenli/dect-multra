%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   Run this file to obtain union of cross-material unitary transforms.
%   This file corresponding to the training process of DECT-CULTRA method.
%   Material images should be prepared for training.
%
%   Zhipeng Li, UM-SJTU Joint Institute, Shanghai Jiao Tong University
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear ; close all; clc
addpath(genpath('../../toolbox'));

%%%%%%%%%%%%%%%%%%%%%%%% load training data here %%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for gam = [0.17]
    for num = [10]
        PatSiz = [8 8 2];  % patch size
        SldDist = 1 * [1 1 1];  % sliding distance

        iter = 2000;  % iteration
        gamma = gam;
        numBlock = num;     % cluster number

        %slice = [75];
        slice = [65 70 75 80 90];
        fprintf('extracting patches...\n')
        patch = [];
        for ii = 1:length(slice)
             for j = 1:2
                if j ==1
                    tmp = phantom(:,:,slice(ii)).*(phantom(:,:,slice(ii)) > 0.9 & phantom(:,:,slice(ii)) < 1.8);
                else
                    tmp = phantom(:,:,slice(ii)).*(phantom(:,:,slice(ii)) > 1.8 & phantom(:,:,slice(ii)) < 2);
                end
                image(:,:,(ii-1)*2+j) = downsample2(tmp, 2); 
             end
             %patch_tmp = volume2patch(single(image(:,:,[(ii-1)*2+1,(ii-1)*2+2]) ), PatSiz, SldDist);
             patch_tmp = im2colstep(single(image(:,:,[(ii-1)*2+1,(ii-1)*2+2]) ), PatSiz, SldDist); 
             patch_tmp = single(patch_tmp);
             patch = [patch patch_tmp];
        end

        patch = double(patch);
        PatNum = size(patch, 2);
        fprintf('Length of training set: %d\n', PatNum);

        IDX = randi(numBlock,PatNum,1);       % Random Initialization

        TransWidth = prod(PatSiz);

        D1 = dctmtx(PatSiz(1));
        D2 = dctmtx(PatSiz(2));
        D3 = dctmtx(PatSiz(3));
        D = kron(kron(D1, D2), D3);  % 3D DCT Initialization. Be careful of the order of D1 D2 D3!
        clear D1 D2 D3

        mOmega = zeros(TransWidth,TransWidth,numBlock, 'double');% must be 'double'!
        for i = 1:numBlock
          mOmega(:,:,i) = D;
        end
        clear D

        perc = zeros(iter,numBlock,'single'); % sparsity (percentage)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Training %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        for j = 1 : iter
          fprintf('iteration = %d:\n', j);
          for k = 1 : numBlock
            patch_k = patch(:, IDX == k); % this way is faster
            %lambada_k = lambada0 * norm(patch_k, 'fro') ^ 2;% lambda_{k} update
            sparseCode = mOmega(:,:,k) * patch_k;
            % hard-thresholding
            sparseCode = sparseCode.*(abs(sparseCode) > gamma);
            perc(j,k) = nnz(sparseCode) / numel(sparseCode)* 100;
            %         fprintf('sparsity(%d) = %g\n', k, perc(j,k));

            % transform update
            if (size(patch_k,2) > 0) % if patch_k is empty, transform will be unchange
                [U,S,V]=svd(patch_k*sparseCode');
                mOmega(:,:,k)=V * U';
            else
              fprintf('patch %g is empty\n', k);
            end
          end
          clear patch_k sparseCode LL2 Q1 sig R B
          fprintf('Cond Number(%d) = %g\n', numBlock, cond(mOmega(:,:,numBlock)));
          fprintf('sparsity(%d) = %g\n', 1, perc(j,1));
          error = zeros(numBlock, PatNum, 'double');
          %%%%%%%%% clustering measure %%%%%%%
          for k = 1 : numBlock
            a1 = mOmega(:,:,k) * patch;
            a0 = a1 .* (abs(a1) > gamma);
            error(k, :) = sum((a1-a0).^2,'double') + gamma^2 * sum(abs(a0) > 0);
          end
          % clustering
          [~, IDX] = min(error, [] ,1); clear  error a1 a0

          show(mOmega)
        end
        %%%%%%%%%%%%%%%%%%%% check cluster-mapping %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %  CluMap = ClusterMap([512 512 2], PatSiz, SldDist, info.IDX, PatNum, numBlock);
        %  figure(55), imshow(CluMap(:,:,2), [1,numBlock]);colorbar
        %  figure(55), im('mid3',permute(CluMap,[2 1 3]), [1,numBlock])
        %  figure(55), im('mid3',permute((CluMap == numBlock).* image,[2 1 3]), [800,1200])

        % [rows,cols] = ind2sub(size(image) - sqrt(l) + 1, idx);  % patch index
        % c = cell(numBlock, 1);          % index of patches in each cluster
        % for k = 1 : numBlock
        %    c{k,1} = find(IDX == k);
        % end
        % [img_rec, clusterMap] = exactClusterK(image, rows, cols, c, numBlock, n, numPatch);
        %
        % imshow(clusterMap,[1,numBlock])
        % segment = clusterVisual(image, clusterMap, numBlock);
        % figure; imshow(segment);

        % CluMap = ClusterMap(ImgSiz, PatSiz, SldDist, IDX, PatNum, numBlock);
        %  figure(55), imshow(CluMap, [1,numBlock]);colorbar
        %  figure(55), im('mid3',permute((CluMap == numBlock).* image,[2 1 3]), [800,1200])

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

        info = struct('SldDist',SldDist,'gamma',gamma,...
          'numBlock',numBlock,'iter',iter,'mOmega',mOmega,'IDX',IDX ,'perc',perc);
       save(sprintf('Unitary_slice5_block%d_SldDist1_iter%d_gamma%2.2g.mat',... 
                numBlock, iter, gamma),'info');
            clearvars -except gam num phantom
            close all; clc
    end
end
%% show pre-learned transform
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
  kb = 7;  counter = 1; ka = 15;
  jy = 1; % control the line interval between different patches

    Ta = zeros((8+jy)*ka+8,(16+jy)*kb+16); % use "ones" for different looking
    for i=1:8+jy:ka*(8+jy)+1
        for j=1:16+jy:(kb*(16+jy))+1
           if(counter<=size(transform,1))    
           Ta(i:i+7,j:j+15) = reshape(transform(counter,:,k),8,16);
           else
               Ta(i:i+7,j:j+15) = 1;
           end
           counter=counter+1;
        end
    end
  
  blank = zeros(size(Ta,1),2);
  Taa = cat(2 , Ta , blank);
  Taaa = cat(2 , Taaa , Taa);
end
figure(66);imagesc(Taaa);colormap('Gray');axis off;axis image; drawnow

end