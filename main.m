%Importing libraries
%http://www.prlab.tudelft.nl/david-tax/
addpath('prtools'); 
addpath('dd_tools');

%Generating one-class banana dataset
x = gendatb([300, 0]);
data = gendatoc(x, []);

%plot original dataset
scatterd(data);
title('Original Dataset');

%Generating pseudo binary data
[targets, outliers] = sds(data);

%plot pseudo SDS binary dataset
figure;
scatterd(outliers, 'r+');
hold on;
scatterd(targets);
title('SDS Pseudo Dataset');
legend('Outliers','Targets', 'Location','northwest');

%Comparing with "Uniform object generation for optimizing one-class classifiers"
%https://dl.acm.org/doi/10.5555/944790.944809
uniform_objects = gendatout(data, 2000);

%plot pseudo Uniform Object binary dataset
figure;
scatterd(uniform_objects, 'r+');
hold on;
scatterd(data);
title('Uniform Object Generation');
legend('Outliers','Targets', 'Location','northwest');

%Optimizing SVDD hyperparameters
params = {};
params{1} = [0 0.05]; %fracrej
params{2} = linspace(0.5, 8, 6);% sigma

%Building grid-search
arg = params{1};
arg = arg(:); % make sure it is a column vector

newarg = params{2};
newarg = newarg(:)'; % make sure it is a row vector

n = length(newarg);
m = size(arg,1);

newarg = repmat(newarg,m,1);
arg = [repmat(arg,n,1) newarg(:)];

arg = num2cell(arg);
nrcomb = size(arg,1);

%number of folds for crossvalidation used only by Uniform Objects
nrfolds = 10;

%Evaluating each combination of parameters
min_err_sds = 2;
min_err_uo = 2;
for i = 1:nrcomb
	thisarg = arg(i,:)
	%Training SVDD on the training set, it can be any one-class classifier
	w = svdd(data, thisarg{:});

	%Testing the classifier on the SDS pseudo binary dataset
	%Error on target class
	err_t_sds = dd_error(targets*w);

	%Error on outlier class
	err_o_sds = dd_error(outliers*w);

	%classifier error
	err_sds = err_t_sds(1) + err_o_sds(2);

	%Selecting classifier with smallest error according to SDS
	if(err_sds < min_err_sds)
		min_err_sds = err_sds
		best_w_sds = w;
	end

	%Testing the classifier on the uniform objects
	%Error on target class
	err_t_uo = zeros(nrfolds, 1);
	I = nrfolds;
	for j=1:nrfolds
		%x - training set, z - test set
		[x,z,I] = dd_crossval(data, I);
		%training
		w1 = svdd(x, thisarg{:});
		%test
		err_xval = dd_error(z, w1);
		err_t_uo(j) = err_xval(1);
	end
	
	%Error on outlier class
	err_o_uo = dd_error(uniform_objects*w);
	
	%classifier error
	err_uo = mean(err_t_uo) + err_o_uo(2);

	%Selecting classifier with smallest error according to Uniform Objects
	if(err_uo < min_err_uo)
		min_err_uo = err_uo
		best_w_uo = w;
	end
end

%Ploting best classifier according to SDS
figure;
scatterd(data);
plotc(best_w_sds);
title('Best classifier according to SDS');

%Ploting best classifier according to Uniform Objects
figure;
scatterd(data);
plotc(best_w_uo);
title('Best classifier according to Uniform Objects');