function [x, cost] = unitary_pwls(x, A, yi, wi, R, chat, iter )

cpu etic

% iterate

% invM = M.^(-1);
npix = size(x, 1)/2;
wij = diag(wi([1 1+npix],1));
yij = cat(2, yi(1:end/2,1), yi(end/2+1:end,1)); yij = yij';
fprintf('minmax(x) = %g , %g\n',min(x) ,max(x)); 

for ii=1 : iter    
     [tmp, denom_tmp] = R.cgrad(x);     
     tmp = [tmp(1:end/2)'; tmp(end/2+1:end)']; 
     
     x = zeros(size(yij));
     parfor jj = 1: size(yij,2)
         denom = A'* wij * A+ diag(denom_tmp(:,jj));
         num = A'* wij * yij(:,jj) + tmp(:,jj);
         x(:,jj) = denom \ num;
     end
     x = x';
     x = cat(1, x(:,1), x(:,2));
 
     if chat
        cost = R.penal(A, x, wi, yi);
     else
        cost = 0;
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