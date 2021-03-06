%DD_F1 compute the F1 score 
%
%   E = DD_F1(X,W)
%   E = DD_F1(X*W)
%   E = X*W*DD_F1
%
% INPUT
%   X    One-class dataset
%   W    One-class classifier
%
% OUTPUT
%   E    F1 performance
%
% DESCRIPTION
% Compute the F1 score of a dataset, defined as:
%           2*precision*recall
%     F1 =  ------------------
%           precision + recall
%
% SEE ALSO
% dd_error, dd_roc

% Copyright: D.M.J. Tax, D.M.J.Tax@prtools.org
% Faculty EWI, Delft University of Technology
% P.O. Box 5031, 2600 GA Delft, The Netherlands

function e = dd_f1(x,w)

% Do it the same as testc:
% When no input arguments are given, we just return an empty mapping:
if nargin==0
	
   e = prmapping(mfilename,'fixed');
   e = setname(e,'F1');
	return

elseif nargin == 1
	% Now we are doing the actual work, our input is a mapped dataset:

	% get the precision and recall:
	[dummy,f] = dd_error(x);
	c = dd_confmat(x);

	% do some checks:
	if ~isfinite(f(1))
		warning('dd_tools:NonfiniteOutputs',...
			'The precision is not finite (all data is classified as outlier)');
	end
	if ~isfinite(f(2))
		warning('dd_tools:NonfiniteOutputs',...
			'The recall is not finite (no target data present?)');
	end
	% and compute F1:
	e(1) = (2*f(1)*f(2))/(f(1)+f(2));
	e(2) = (2*f(3)*f(4))/(f(3)+f(4));
	e(3) = (c(1)*c(4)-c(3)*c(2))/sqrt((c(1)+c(3))*(c(1)+c(2))*(c(4)+c(3))*(c(4)+c(2)));

	if(isnan(e(1)))
		e(1) = 0;
	end

	if(isnan(e(2)))
		e(2) = 0;
	end

	if(isnan(e(3)))
		e(3) = 0;
	end

else

	ismapping(w);
	istrained(w);

	e = feval(mfilename,x*w);

end

return

