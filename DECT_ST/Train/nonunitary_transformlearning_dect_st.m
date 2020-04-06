%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear ; close all; clc
addpath(genpath('~/Desktop/data/2Dxcat'));
addpath(genpath('~/Desktop/toolbox'));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Initialization  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dir = ['/home/zhipeng/Desktop/Poly_DECT_ST_clinical_v2'];
load([dir '/phantom.mat']); %units g/cm^3
%phantom = phantom_wb_cm;clear phantom_wb_cm
%phantom = rot90(phantom,3);

pix2cm = 1/0.05;
cm2HU = 1000/0.2;

water = 1;

       l = 64; % patch size
  stride = 1;  % overlapping stride      
lambada0 = 31;  % 0.031 is fine 
    iter = 2000; % iteration
   gamma = 0.12;   

if water
    phantom = phantom.*(phantom > 0.9 & phantom < 1.8);
else
    phantom = phantom.*(phantom > 1.8 & phantom < 2);
end
% btype = {'water','bone'};
% LL = length(btype);
% for ll = 1 : LL
%     density{ll} = xray_read_dens(btype{ll});
% end
% if water
%     phantom = density{1} * phantom;
% else
%     phantom = density{2} * phantom;
% end
%imshow(phantom(:,:,end/2),[800 1200]/cm2HU);  
patch=[];
for ii = [65 70 75 80 90]  % training data
%  for ii = 45     % testing data
   image = downsample2(phantom(:, :, ii), 2); 
% The Mathworks 'im2col' is quicker but only for stride 1.    
%    patch_tmp = im2col(image, sqrt(l) * [1 1], 'sliding'); 
%     [patch_tmp, ~] = image2patch(image,sqrt(l) * [1 1], stride);
   patch_tmp = im2colstep(single(image), sqrt(l)*[1 1], stride*[1 1]);   
   patch = [patch patch_tmp];
end
 
mOmega = kron(dctmtx(sqrt(l)),dctmtx(sqrt(l)));% DCT Initialization
lambada = lambada0 * norm(patch,'fro')^2;
    
%[U,S,V] = svd((patch*patch') + (0.5*lambada*eye(l))); 
[U,S,V] = svd((patch*patch') + lambada*eye(l));
LL2 = (inv(U*(S^(1/2))*V'));
    
perc=[]; condTransform=[]; % sparsity(percentage) 
for j=1:iter
    fprintf('iteration = %d:\n', j);
    % sparse coding
    sparseCode = mOmega * patch;
    sparseCode = sparseCode.*(abs(sparseCode) > gamma);
    perc(j) = nnz(sparseCode)/ numel(sparseCode) * 100;
    fprintf('sparsity  = %g:\n', perc(j));
    % transform update
    [Q1,Si,R] = svd(LL2*patch*sparseCode');
    sig = diag(Si);
    gamm = (1/2)*(sig + (sqrt((sig.^2) + 2*lambada)));
    B = R*(diag(gamm))*Q1';
    mOmega = B*(LL2);
  
    show(mOmega);
   
%     condTrans(j) = cond(transform); 
end
save(sprintf('water_den_l%d_stride%d_l0%d_iter%d_gamma%2.2g.mat', l, stride,lambada0, iter, gamma),'mOmega');     
% save('mOmega.mat','mOmega')
% save('perc.mat','perc')

% check convergency  
figure(); plot(perc);
xlabel('Number of Iteration','fontsize',18)
ylabel('Sparity ( % )','fontsize',18) 
% check condition number
% condTrans = cond(transform); 
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
figure;imagesc(Ta);colormap('Gray');axis off;axis image; 
drawnow;
end   
    