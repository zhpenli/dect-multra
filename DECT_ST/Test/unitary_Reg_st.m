classdef unitary_Reg_st < handle

 properties
   mMask; % the mask matrix
   PatSiz;   % [rows cols thickness] 1x2 vector patch size
   ImgSiz; % [rows cols thickness] 1x2 vector
   SldDist; % sliding distance
   beta1; %parameter controls the balance between noise and resolution
   gamma1; % threshold
   mOmega1; % transform matrix 
   beta2;
   gamma2;
   mOmega2;
   mSpa1; % the matrix of sparse code 
   mSpa2;
   rstSpa;
 end
 
 methods  
function obj = unitary_Reg_st(mask, PatSiz, ImgSiz, SldDist, beta1, beta2,...
     gamma1, gamma2, mOmega1, mOmega2)
     obj.mMask = mask;
     obj.PatSiz = PatSiz; 
     obj.ImgSiz = ImgSiz;
     obj.SldDist = SldDist;
     obj.beta1 = beta1;
     obj.gamma1 = gamma1;
     obj.mOmega1 = mOmega1;
     obj.beta2 = beta2;
     obj.gamma2 = gamma2;
     obj.mOmega2 = mOmega2;
     obj.rstSpa = true;
 end
     
 function cost = penal(obj, A, x, wi, yi)
     % data fidelity
     npix = size(x,1)/2;
     xc = cat(2, x(1:end/2,1), x(end/2+1:end,1)); xc =xc'; 
     yc = cat(2, yi(1:end/2,1), yi(end/2+1:end,1)); yc =yc';
     wij = diag(wi([1 1+npix],1));
     df = .5 * sum(col(wij * ((A * xc - yc).^2)), 'double');
    % fprintf('df = %g\n', df); 
     
     x1 = embed(x(1:1:end/2,1), obj.mMask);  
     x1 = single(x1);
     mPat1 = im2colstep(x1, obj.PatSiz, obj.SldDist); clear x1;
     mPat1 = single(mPat1);
%      mPat = volume2patch(x, obj.PatSiz, obj.SldDist); clear x;
     mCod1 = obj.mOmega1 * mPat1;  clear mPat1;        
     % sparsity error
     spa_err1 = obj.beta1 *  sum(col(mCod1 - obj.mSpa1).^2); clear mCod1;
     
     x2 = embed(x(end/2+1:1:end,1), obj.mMask);  
     x2 = single(x2);
     mPat2 = im2colstep(x2, obj.PatSiz, obj.SldDist); clear x2;
     mPat2 = single(mPat2);
%      mPat = volume2patch(x, obj.PatSiz, obj.SldDist); clear x;
     mCod2 = obj.mOmega2 * mPat2;  clear mPat2;        
     % sparsity error
     spa_err2 = obj.beta2 *  sum(col(mCod2 - obj.mSpa2).^2); clear mCod2;
     spa_err = spa_err1 + spa_err2;     
     %fprintf('se = %g\n', spa_err); 
     
     tmp1 = sum(col(abs(obj.mSpa1) > 0));  % l0norm
     spa1 = obj.beta1 * (obj.gamma1^2 * tmp1 );
     tmp2 = sum(col(abs(obj.mSpa2) > 0));  % l0norm
     spa2 = obj.beta2 * (obj.gamma2^2 * tmp2 );
     spa = spa1 + spa2;
     %fprintf('sp = %g\n', spa); 
     
     perc1 = tmp1 / numel(obj.mSpa1) * 100; clear tmp1;
     perc2 = tmp2 / numel(obj.mSpa2) * 100; clear tmp2;
     fprintf('perc1 = %g, perc2 = %g\n', perc1, perc2); 
     
     cost_val = df + spa_err + spa;
     cost=[]; cost(1)= cost_val; cost(2)= df; 
     cost(3)= spa_err; cost(4)= spa; cost(5) = perc1; cost(6) = perc2;
     fprintf('cost_val = %g\n', cost_val); 
end
   
function grad = cgrad(obj, x)
     x1 = embed(x(1:1:end/2,1), obj.mMask);
     x1 = single(x1);
     x2 = embed(x(end/2+1:1:end,1), obj.mMask);
     x2 = single(x2);
     if obj.rstSpa
        % fprintf('Sparse Coding Step...\n')
         mPatx1 = im2colstep(x1, obj.PatSiz, obj.SldDist); clear x1;
         mPatx1 = single(mPatx1);
         mCodx1 = obj.mOmega1 * mPatx1; clear mPatx1;
         obj.mSpa1 = mCodx1 .* (abs(mCodx1) > obj.gamma1); %calculate z_{1j}
         diff1 = obj.mOmega1' * (obj.mSpa1); clear mCodu1; 
         
         mPatx2 = im2colstep(x2, obj.PatSiz, obj.SldDist); clear x2;
         mPatx2 = single(mPatx2);
         mCodx2 = obj.mOmega2 * mPatx2; clear mPatx2;
         obj.mSpa2 = mCodx2 .* (abs(mCodx2) > obj.gamma2); %calculate z_{2j}
         diff2 = obj.mOmega2' * (obj.mSpa2); clear mCodu2;  
         obj.rstSpa = false;
     else
         diff1 = obj.mOmega1' * (obj.mSpa1);
         diff2 = obj.mOmega2' * (obj.mSpa2);
     end
     diff1 = single(diff1);
     grad1 = 2 * obj.beta1 .* col2imstep(diff1, obj.ImgSiz, obj.PatSiz, obj.SldDist); 
     grad1 = single(grad1);
     grad1 = grad1(obj.mMask);  clear diff1;
         
     diff2 = single(diff2);
     grad2 = 2 * obj.beta2 .* col2imstep(diff2, obj.ImgSiz, obj.PatSiz, obj.SldDist); 
     grad2 = single(grad2);
     grad2 = grad2(obj.mMask);  clear diff2;
     
     grad = cat(1, grad1, grad2);
     
     
end

 function nextOuterIter(obj)
    % set the flag of updating SparseCode
    obj.rstSpa = true;
 end  
       
end
    
end