classdef unitary_Reg_CULTRA < handle

 properties
   mMask;    % the mask matrix
   ImgSiz;   % image size   
   PatSiz;   % patch size
   SldDist;  % sliding distance
   beta;
   gamma;    % threshold
   mOmega;   % transform matrix 
   numBlock; % the number of square transforms
   vIdx;     % the patch index decides which transform belongs to 
   isSpa;    % the flag of sparse code update 
   isClu;    % the flag of clustering update
   CluInt;   % the number of clustering interval
   mSpa;     % the matrix of sparse code    
 end
 
 methods  
  function obj = unitary_Reg_CULTRA(mask, ImgSiz, PatSiz, SldDist, beta, gamma, mOmega, numBlock, CluInt)
     obj.mMask = mask;
     obj.PatSiz = PatSiz; 
     obj.ImgSiz = ImgSiz;
     obj.SldDist = SldDist;
     obj.beta = beta;
     obj.gamma = gamma;
     obj.mOmega = mOmega;
     obj.numBlock = numBlock;
     obj.isSpa = true; 
     obj.isClu = CluInt;
     obj.CluInt = CluInt;
     obj.vIdx = [];
   end
     
 function cost = penal(obj, A, x, wi, yi)
     % data fidelity
     npix = size(x,1)/2;
     xc = cat(2, x(1:end/2,1), x(end/2+1:end,1)); xc =xc'; 
     yc = cat(2, yi(1:end/2,1), yi(end/2+1:end,1)); yc =yc';
     wij = diag(wi([1 1+npix],1));
     df = .5 * sum(col(wij * ((A * xc - yc).^2)), 'double');
     fprintf('df = %g ', df); 
     
     x1 = embed(x(1:1:end/2,1), obj.mMask);  
     x1 = single(x1);
     x2 = embed(x(end/2+1:1:end,1), obj.mMask);  
     x2 = single(x2);
     xdim = cat(3, x1, x2);
     mPat = im2colstep(xdim, obj.PatSiz, obj.SldDist); clear x1 x2;
     mPat = single(mPat);
     mCod = zeros(size(mPat), 'single'); 
      for k = 1 : obj.numBlock
          tmp = obj.vIdx==k; 
          mCod(:,tmp) = obj.mOmega(:,:,k) * mPat(:,tmp) ;         
      end       
      clear mPat    
      % sparsity error
       spa_err = obj.beta * sum( ( col(mCod - obj.mSpa) ).^2,1) ; clear mCod;
       fprintf('se = %g  ', spa_err);
       spa = obj.beta * obj.gamma^2 * nnz(obj.mSpa);% l0norm
       fprintf('sp = %g  ', spa);     

      cost_val = df + spa_err + spa;
      fprintf('costval = %g\n  ', cost_val); 
      cost=[]; cost(1)= cost_val; cost(2)= df; 
      cost(3)= spa_err; cost(4)= spa;
end
   
function grad = cgrad(obj, x)
     x1 = embed(x(1:1:end/2,1), obj.mMask);
     x1 = single(x1);
     x2 = embed(x(end/2+1:1:end,1), obj.mMask);
     x2 = single(x2);
   
     mPatx = im2colstep(cat(3, x1, x2), obj.PatSiz, obj.SldDist); clear x1 x2;
      %%%%%%%%%%%%% cluster index update %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      if(obj.isClu == obj.CluInt)    
         numPatch = size(mPatx, 2);  
         error = zeros(obj.numBlock, numPatch, 'double');
         %%%%%%%%%%%%%% clustering measure %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
         for k = 1 : obj.numBlock
             a1 = obj.mOmega(:,:,k) * mPatx;
             a0 = a1 .* (abs(a1) >= obj.gamma);  
             error(k,:) = sum((a1 - a0).^2,'double') + obj.gamma^2 *sum(abs(a0)>0);
         end
         clear a0 a1;
         %%%%%%%%%%%%%% clustering %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
         if(isempty(obj.vIdx))          
            obj.vIdx = ones(1, numPatch);
         end          
         [~, obj.vIdx] = min(error, [] ,1);  clear error;  
       
         obj.isClu = 0; % reset clustering counter
      end   
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
      diff = zeros(size(mPatx), 'single');          
      if(obj.isSpa)    
         mCodx = zeros(size(mPatx), 'single');
         for k = 1 : obj.numBlock
             tmp = obj.vIdx==k;  
             mCodx(:,tmp) = obj.mOmega(:,:,k) * mPatx(:,tmp) ;      
         end       
         clear mPatu mPatx 
         obj.mSpa = mCodx .* (abs(mCodx) >= obj.gamma); 
         
         for k = 1:obj.numBlock 
             tmp = obj.vIdx==k;             
             diff(:,tmp) = obj.mOmega(:,:,k)' * obj.mSpa(:,tmp);
         end   
         obj.isSpa = false;  % close the flag of sparse code update       
      else     
         for k = 1:obj.numBlock  
             tmp = obj.vIdx==k;   
             diff(:,tmp) = obj.mOmega(:,:,k)' * obj.mSpa(:,tmp);
         end 
         clear mPat         
      end
      grad = 2 * obj.beta .* col2imstep(single(diff), obj.ImgSiz, obj.PatSiz, obj.SldDist); 
      grad1 = grad(:,:,1); grad2 = grad(:,:,2);
      grad = [grad1(obj.mMask); grad2(obj.mMask)];
     
end

 function [perc, vIdx] = nextOuterIter(obj)
      obj.isClu = obj.isClu + 1;
      vIdx = obj.vIdx;   
      obj.isSpa = true; % open the flag of updating sparse code
      % sparsity check
      perc = nnz(obj.mSpa) / numel(obj.mSpa) * 100;
 end
 
 function SparseCode = last_spa(obj)
    SparseCode = obj.mSpa;
 end
       
end
    
end