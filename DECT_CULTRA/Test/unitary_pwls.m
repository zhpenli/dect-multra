function [x, cost] = unitary_pwls(x, A, yi, wi, R, M, chat, iter )

cpu etic

% iterate
npix = size(x, 1)/2;
M = diag(M([1 1+npix],1));
wij = diag(wi([1 1+npix],1));
yij = cat(2, yi(1:end/2,1), yi(end/2+1:end,1)); yij = yij';
fprintf('minmax(x) = %g , %g\n',min(x) ,max(x)); 

for ii=1 : iter 
     tmp = R.cgrad(x);     
     tmp = [tmp(1:end/2)'; tmp(end/2+1:end)'];
     denom = A'* wij * A + M ;
     num = A'* wij * yij + tmp;
     x = denom \ num;
     x = x';
     x = cat(1, x(:,1), x(:,2));

%      theta_old = theta;
%      x_old = x;    
%      denom = sum(A'* wij * A, 2) + diag(M) ;
%      cgrad = R.cgrad(u,x);
%      u = cat(2, u(1:end/2,1), u(end/2+1:end,1)); u =u'; 
%      cgrad = cat(2, cgrad(1:end/2,1), cgrad(end/2+1:end,1)); cgrad =cgrad';
%      num = A'* wij *(A*u - yij) + cgrad;
%      x = u - num ./ repmat(denom,1 ,npix);
%      x = x';
%      x = cat(1, x(:,1), x(:,2));
    
   % x = max(x,0);   
  if chat
    % calculate the cost value for each outer iteration
     cost = R.penal(A, x, wi, yi);
  else
     cost = 0;
 %    SpaCode = R.last_spa();
  end 
   
end

 

% default user function.
% using this evalin('caller', ...) trick, one can compute anything of interest
function out = userfun_default(x, varargin)
chat = evalin('caller', 'arg.chat');
if chat
%	x = evalin('caller', 'x');
	printm('minmax(x) = %g %g', min(x), max(x))
end
out = cpu('etoc');