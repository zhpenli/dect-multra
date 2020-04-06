classdef unitary_Reg_MULTRA < handle

 properties
   mMask;    % the mask matrix
   ImgSiz1;   % image size  
   ImgSiz2;
   PatSiz1;   % patch size
   SldDist1;  % sliding distance
   PatSiz2;
   SldDist2;
   beta;
   gamma;    % threshold
   mOmega1;   % transform matrix
   mOmega2;
   mOmega3;
   numBlock; % the number of square transforms
   vIdx1;     % the patch index decides which transform belongs to 
   vIdx2;
   vIdx3;
   isSpa;    % the flag of sparse code update 
   isClu;    % the flag of clustering update
   CluInt;   % the number of clustering interval
   mSpa1;     % the matrix of sparse code 
   mSpa2;
   mSpa3;
   IDX_water;
   IDX_cross;
   IDX_bone;
 end
 
 methods  
  function obj = unitary_Reg_MULTRA(mask, ImgSiz1, ImgSiz2, PatSiz1, SldDist1, PatSiz2, SldDist2, ...
          beta, gamma, mOmega1, mOmega2, mOmega3, numBlock, CluInt)
     obj.mMask = mask;
     obj.PatSiz1 = PatSiz1; 
     obj.ImgSiz1 = ImgSiz1;
     obj.ImgSiz2 = ImgSiz2;
     obj.SldDist1 = SldDist1;
     obj.PatSiz2 = PatSiz2; 
     obj.SldDist2 = SldDist2;
     obj.beta = beta;
     obj.gamma = gamma;
     obj.mOmega1 = mOmega1;
     obj.mOmega2 = mOmega2;
     obj.mOmega3 = mOmega3;
     obj.numBlock = numBlock;
     obj.isSpa = true; 
     obj.isClu = CluInt;
     obj.CluInt = CluInt;
     obj.vIdx1 = [];
     obj.vIdx2 = [];
     obj.vIdx3 = [];
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
     mPat = im2colstep(xdim, obj.PatSiz1, obj.SldDist1); clear x1 x2;
     mPat = single(mPat);
     mPat1 = mPat(1:end/2,obj.IDX_water); 
     mPat2 = mPat(:,obj.IDX_cross);
     mPat3 = mPat(end/2+1:end,obj.IDX_bone);
     for ii = 1:3
         if ii == 1
              mCod1 = zeros(size(mPat,1)/2, size(obj.IDX_water,2), 'single'); 
         elseif ii == 2
              mCod2 = zeros(size(mPat,1), size(obj.IDX_cross,2), 'single'); 
         elseif ii == 3
              mCod3 = zeros(size(mPat,1)/2, size(obj.IDX_bone,2), 'single'); 
         end
          for k = 1 : obj.numBlock(1,ii)
              eval([ 'tmp = obj.vIdx',num2str(ii), '==k;' ]); 
              eval([ 'mCod',num2str(ii),'(:,tmp) = obj.mOmega',num2str(ii),'(:,:,k) * mPat',num2str(ii),'(:,tmp);']);         
          end     
     end
      clear mPat    
      % sparsity error
       spa_err = obj.beta * ( sum( ( col(mCod1 - obj.mSpa1) ).^2,1) +sum( ( col(mCod2 - obj.mSpa2) ).^2,1)...
           +sum( ( col(mCod3 - obj.mSpa3) ).^2,1)); clear mCod;
       fprintf('se = %g  ', spa_err);
       spa = obj.beta * (obj.gamma(1,1)^2*nnz(obj.mSpa1)+obj.gamma(1,2)^2 *...
           nnz(obj.mSpa2)+obj.gamma(1,3)^2 *nnz(obj.mSpa3));% l0norm
       fprintf('sp = %g  ', spa);     

      cost_val = df + spa_err + spa;
      fprintf('costval = %g\n  ', cost_val); 
      cost=[]; cost(1)= cost_val; cost(2)= df; 
      cost(3)= spa_err; cost(4)= spa;
end
   
function [grad,denom] = cgrad(obj, x)
     x1 = embed(x(1:1:end/2,1), obj.mMask);
     x1 = single(x1);
     x2 = embed(x(end/2+1:1:end,1), obj.mMask);
     x2 = single(x2);
  
     mPatx = im2colstep(cat(3, x1, x2), obj.PatSiz1, obj.SldDist1); clear x1 x2;
      %%%%%%%%%%%%% cluster index update %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      if(obj.isClu == obj.CluInt)    
         numPatch = size(mPatx, 2);  
         %%%%%%%%%%%%%% clustering measure %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
         for ii = 1:3
             mPatx1 = mPatx(1:end/2,:); mPatx2 = mPatx; mPatx3 = mPatx(end/2+1:end,:);
             eval([ 'error', num2str(ii), '= zeros(obj.numBlock(1,ii), numPatch);' ]);
             for k = 1 : obj.numBlock(1,ii)
                 if ii == 1
                     a1 = obj.mOmega1(:,:,k) * mPatx1;
                     a0 = a1 .* (abs(a1) >= obj.gamma(1,1));  
                     error1(k,:) = sum((a1 - a0).^2,'double') + obj.gamma(1,1)^2 *sum(abs(a0)>0);
                 elseif ii == 2
                     a1 = obj.mOmega2(:,:,k) * mPatx2;
                     a0 = a1 .* (abs(a1) >= obj.gamma(1,2));  
                     error2(k,:) = sum((a1 - a0).^2,'double') + obj.gamma(1,2)^2 *sum(abs(a0)>0);
                 elseif ii == 3
                     a1 = obj.mOmega3(:,:,k) * mPatx3;
                     a0 = a1 .* (abs(a1) >= obj.gamma(1,3));  
                     error3(k,:) = sum((a1 - a0).^2,'double') + obj.gamma(1,3)^2 *sum(abs(a0)>0);
                 end
             end
             clear a0 a1;
             %%%%%%%%%%%%%% clustering %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
             if(isempty(eval( ['obj.vIdx',num2str(ii)] )))          
                eval( ['obj.vIdx',num2str(ii), '= ones(1, numPatch);' ]);
             end 
         end

            error1_min = min(error1,[],1);
            error2_min = min(error2,[],1);
            error3_min = min(error3,[],1);
%             obj.IDX_cross = find(error2_min < (error1_min + error3_min));
%             obj.IDX_water = setdiff( find(error1_min <= error3_min), obj.IDX_cross );
%             obj.IDX_bone = setdiff( find(error3_min < error1_min ), obj.IDX_cross );
            obj.IDX_cross = find(error2_min < (error1_min + error3_min));
            obj.IDX_water = find(error2_min >= (error1_min + error3_min));
            obj.IDX_bone = find(error2_min >= (error1_min + error3_min));
            
            [~,obj.vIdx1] = min(error1(:,obj.IDX_water),[],1);
            [~,obj.vIdx2] = min(error2(:,obj.IDX_cross),[],1);
            [~,obj.vIdx3] = min(error3(:,obj.IDX_bone),[],1);
        
            Patx_tmp = zeros(size(mPatx));
            Patx_tmp(:,obj.IDX_water) = 2 * obj.beta(1);
            Patx_tmp(:,obj.IDX_cross) = 2 * obj.beta(2);
            denom = col2imstep(single(Patx_tmp), obj.ImgSiz1, obj.PatSiz1, obj.SldDist1);
            denom1 = denom(:,:,1); denom2 = denom(:,:,2);
            denom = cat(1, denom1(obj.mMask)', denom2(obj.mMask)');

%         error13 = error1 + error3;
%         error13_min = min(error13,[],1);  error2_min = min(error2,[],1); 
%         obj.IDX_cross = find(error2_min < error13_min);
%         obj.IDX_water = find(error2_min >= error13_min);
%         obj.IDX_bone = find(error2_min >= error13_min);
%         [~,obj.vIdx1] = min(error13(:,obj.IDX_water),[],1);
%         [~,obj.vIdx2] = min(error2(:,obj.IDX_cross),[],1);
%         [~,obj.vIdx3] = min(error13(:,obj.IDX_bone),[],1);
       
         obj.isClu = 0; % reset clustering counter
      end    
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
      clear mPatx1 mPatx2 mPatx3
      if(obj.isSpa) 
          for ii = 1:3
              if ii == 1
                 mPatx1 = mPatx(1:end/2,obj.IDX_water);
                 mCodx1 = zeros(size(mPatx1), 'single');
              elseif ii == 2
                 mPatx2 = mPatx(:,obj.IDX_cross);
                 mCodx2 = zeros(size(mPatx2), 'single');
              elseif ii == 3
                 mPatx3 = mPatx(end/2+1:end,obj.IDX_bone);
                 mCodx3 = zeros(size(mPatx3), 'single');
              end
              
              for k = 1 : obj.numBlock(1,ii)
                  tmp = eval( [ 'obj.vIdx',num2str(ii),'==k;' ]);  
                  eval([ 'mCodx',num2str(ii), '(:,tmp) = obj.mOmega',num2str(ii),'(:,:,k) * mPatx',num2str(ii),'(:,tmp);' ]);     
              end       
              clear mPatu1 mPatu2 mPatu3 mPatx1 mPatx2 mPatx3 
              
              if ii == 1
                  obj.mSpa1 = mCodx1 .* (abs(mCodx1) >= obj.gamma(1,1)); 
              elseif ii == 2
                  obj.mSpa2 = mCodx2 .* (abs(mCodx2) >= obj.gamma(1,2)); 
              elseif ii == 3
                  obj.mSpa3 = mCodx3 .* (abs(mCodx3) >= obj.gamma(1,3)); 
              end
              
              for k = 1:obj.numBlock(1,ii) 
                  tmp = eval([ 'obj.vIdx',num2str(ii),'==k;' ]); 
                  if ii == 1
                      diff1(:,tmp) = obj.mOmega1(:,:,k)' *  obj.mSpa1(:,tmp);
                  elseif ii == 2
                      diff2(:,tmp) = obj.mOmega2(:,:,k)' *  obj.mSpa2(:,tmp);
                  elseif ii == 3
                      diff3(:,tmp) = obj.mOmega3(:,:,k)' *  obj.mSpa3(:,tmp);
                  end
              end             
          end
          obj.isSpa = false;  % close the flag of sparse code update       
      else
          for ii = 1:3
             for k = 1:obj.numBlock(1,ii)  
                 tmp = eval([ 'obj.vIdx',num2str(ii),'==k;' ]); 
                 if ii == 1
                      diff1(:,tmp) = obj.mOmega1(:,:,k)' *  obj.mSpa1(:,tmp);
                 elseif ii == 2
                      diff2(:,tmp) = obj.mOmega2(:,:,k)' *  obj.mSpa2(:,tmp);
                 elseif ii == 3
                      diff3(:,tmp) = obj.mOmega3(:,:,k)' *  obj.mSpa3(:,tmp);
                 end
             end 
          end
         clear mPat         
      end
      diff1_tmp = zeros(size(diff1,1),numPatch);
      diff2_tmp = zeros(size(diff2,1),numPatch);
      diff3_tmp = zeros(size(diff3,1),numPatch);
      
      diff1_tmp(:,obj.IDX_water) = diff1; 
      diff2_tmp(:,obj.IDX_cross) = diff2;
      diff3_tmp(:,obj.IDX_bone) = diff3;
      
      grad1 = 2 * obj.beta(1) .* col2imstep(single(diff1_tmp), obj.ImgSiz2, obj.PatSiz2, obj.SldDist2);
      grad2 = 2 * obj.beta(2) .* col2imstep(single(diff2_tmp), obj.ImgSiz1, obj.PatSiz1, obj.SldDist1);
      grad3 = 2 * obj.beta(1) .* col2imstep(single(diff3_tmp), obj.ImgSiz2, obj.PatSiz2, obj.SldDist2);
      
      grad2a = grad2(:,:,1); grad2b = grad2(:,:,2);
      grad = [grad1(obj.mMask); grad3(obj.mMask)]+[grad2a(obj.mMask); grad2b(obj.mMask)];
%       grad1 = 2 * obj.beta .* col2imstep(single(diff1), obj.ImgSiz2, obj.PatSiz2, obj.SldDist2);
%       grad3 = 2 * obj.beta .* col2imstep(single(diff3), obj.ImgSiz2, obj.PatSiz2, obj.SldDist2);
%       grad2 = 2 * obj.beta .* col2imstep(single(diff2), obj.ImgSiz1, obj.PatSiz1, obj.SldDist1);
%       
%       grad2_tmpa = grad2(:,:,1); grad2_tmpb = grad2(:,:,2); 
%       grad = [grad1(obj.mMask); grad3(obj.mMask)]+[grad2_tmpa(obj.mMask); grad2_tmpb(obj.mMask)];
     
end

 function [perc, vIdx1, vIdx2, vIdx3, IDX_water, IDX_cross, IDX_bone] = nextOuterIter(obj)
      obj.isClu = obj.isClu + 1;
      vIdx1 = obj.vIdx1; 
      vIdx2 = obj.vIdx2;
      vIdx3 = obj.vIdx3;
      IDX_water = obj.IDX_water;
      IDX_cross = obj.IDX_cross;
      IDX_bone = obj.IDX_bone;
      obj.isSpa = true; % open the flag of updating sparse code
      % sparsity check
      perc(1,1) = nnz(obj.mSpa1) / numel(obj.mSpa1) * 100;
      perc(2,1) = nnz(obj.mSpa2) / numel(obj.mSpa2) * 100;
      perc(3,1) = nnz(obj.mSpa3) / numel(obj.mSpa3) * 100;
 end
 
 function SparseCode = last_spa(obj)
    SparseCode = obj.mSpa;
 end
       
end
    
end