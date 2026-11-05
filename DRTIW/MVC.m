function [B,B_v,H_v,H,w,Z,obj] = MVC(X,alpha,beta,clu_num,maxIter)
    sigma = 2.0;
    epilision = 1e-6;

	view_num = max(size(X));
	dims = zeros(1,view_num);
	for v = 1:view_num
		[dims(v),n] = size(X{v});
	end
	
	B = cell(1,view_num);
	B_v = cell(1,view_num);
	H_v = cell(1,view_num);
	H = zeros(clu_num,n);
	w = ones(view_num,1)./view_num; %初始化权重向量w
	Z = cell(view_num,view_num);
	
	for v = 1:view_num
		B{v} = orth(rand(dims(v),clu_num)); %正交初始化主要基底B
	end
	
	for v = 1:view_num %初始化稀疏的Bv矩阵
		tem = sprand(dims(v),clu_num,0.05);
		B_v{v} = full(tem);
        % B_v{v} = zeros(dims(v),clu_num);  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	end
	clear tem;
	
	for i = 1:view_num
		for j = 1:view_num
			if i~=j
				Z{i,j} = eye(n); %初始化潜在表达的自表达矩阵Z
			else
				Z{i,j} = -1;
			end
		end
	end
	
	% for v = 1:view_num %随机的方式初始化H_v 和H
	% 	H_v{v} = rand(clu_num,n);
	% end
	% H = rand(clu_num,n);
	

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	for v = 1:view_num  %聚类的方式初始化潜在表达矩阵
		data_v = X{v}';
		Dis = squareform(pdist(data_v,'squaredeuclidean'));
		Dis = -1.*Dis./sigma;
		S_temp = exp(Dis);
		rowSums = sum(S_temp,2);
		rowSum_v = repmat(rowSums,1,size(S_temp,2));
		S_temp = S_temp./rowSum_v;
		S_v{v} = (S_temp+S_temp')/2;

		L = SpectralClustering(S_v{v},clu_num);
		Temp = zeros(clu_num,n);
		for i = 1:n
			Temp(L(i),i) = 1;
		end

		for j = 1:clu_num
			Temp(j,:) = Temp(j,:)./sqrt(sum(Temp(j,:)));
		end
		H_v{v} = Temp;
	end
	clear Dis S_temp rowSum_v rowSums Temp L;

	XX = X{1};
	for v = 2:view_num
		XX = [XX ;X{v}];
	end
	data_v = XX';
	Dis = squareform(pdist(data_v,'squaredeuclidean'));
	Dis = -1.*Dis./sigma;
	S_temp = exp(Dis);
	rowSums = sum(S_temp,2);
	rowSum_v = repmat(rowSums,1,size(S_temp,2));
	S_temp = S_temp./rowSum_v;
	S = (S_temp+S_temp')/2;
	Y_pre = SpectralClustering(S,clu_num);
	Temp = zeros(clu_num,n);
	for i = 1:n
		Temp(Y_pre(i),i) = 1;
	end

	for j = 1:clu_num
		Temp(j,:) = Temp(j,:)./sqrt(sum(Temp(j,:)));
	end
	H = Temp;
	clear XX data_v Dis S_temp rowSum_v rowSums S_temp Y_pre Temp;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	temsum = 0;
	for i = 1:view_num
		for j= 1:view_num
			if i ~= j
				temsum = temsum + norm(Z{i,j},'fro')^2;
			end
		end
	end
	%mu = 0.5*temsum/view_num;
    %mu = max(0.5*temsum/view_num, 0.1);   % 添加下界 0.1
    mu = temsum/view_num;


	for iter = 1:maxIter
		%%update B %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%update B
		for v = 1:view_num
			TemS = X{v} - B_v{v}*H_v{v};
			J = TemS*H_v{v}';
			[U,~,V] = svd(J,'econ');
			B{v} = U*V';
		end
		clear TemS J U V;
		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% B updated
		
		%% update B_v %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%update B_v
		for v = 1:view_num
			%% 优化传参的时候需要将B_v转置
			A = X{v} - B{v}*H_v{v};
			%[tem,~] = FISTA_matrix(H_v{v}',A',(1-w(v)),20,0.0001); %%最大迭代次数设置为了20
            [tem,~] = FISTA_matrix(H_v{v}', A', 1, 20, 0.0001);
			B_v{v} = tem';
		end
		clear tem A;
		%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% B_v updated
		
		%%update H %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% update H
		tem = zeros(clu_num,n);
		for v = 1:view_num
			tem = tem + w(v)*H_v{v};
		end
		H = tem;
		clear tem;
		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% H updated
		
		%%update H_v %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		for v = 1:view_num
			T = B{v} + B_v{v};
			left = T'*T + (alpha*w(v)+view_num-1)*eye(clu_num);
			right = zeros(n,n);
			for i = 1:view_num
				if i~=v
					right = right + Z{i,v}*Z{i,v}';
				end
			end
			Erig = T'*X{v} + alpha*w(v)*H;
			for j = 1:view_num
				if j ~= v
					Erig = Erig + H_v{j}*Z{v,j};
				end
			end
			for i = 1:view_num
				if i~=v
					Erig = Erig + H_v{i}*Z{i,v}';
				end
            end
            % try
			H_v{v} = lyap(left, right, -Erig);
            % end
		end
		clear left right Erig T;
		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% H_v updated
		
		%%update Z %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        inner_max_iter = 5;        
        inner_tol = 1e-4;           
        
        for i = 1:view_num
            for j = 1:view_num
                if i ~= j
                    P = Z{i,j}';
                    M = H_v{j}' * H_v{j};
                    C = H_v{i}' * H_v{j};
                    iter_inner = 0;
                    converged = false;
                    while iter_inner < inner_max_iter && ~converged
                        P_old = P;
                        row_norms = sqrt(sum(P.^2, 2) + epilision);
                        Lambda = diag(0.5 ./ (row_norms + 1e-6));
                        N = Lambda + (beta / (mu * (w(j) + epilision))) * eye(n);
                        P = lyap(N, M, -C);
                        rel_change = norm(P - P_old, 'fro') / (norm(P_old, 'fro') + eps);
                        if rel_change < inner_tol
                            converged = true;
                        end
                        iter_inner = iter_inner + 1;
                    end
                    Z{i,j} = P';
                end
            end
        end
		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Z updated
		
		%%update w %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        a = zeros(view_num,1);
        b = zeros(view_num,1);
        for i = 1:view_num
            a(i) = alpha * norm(H - H_v{i}, 'fro')^2;
            b(i) = 0;
            for j = 1:view_num
                if i ~= j
                    b(i) = b(i) + norm(Z{j,i}, 'fro')^2;
                end
            end
        end
        w = optimize_w_analytic(a, b, mu, beta);
        w = max(w, 0);
        w = w / sum(w);		
		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%w updated
		

		
		
		tem_obj1 = 0;
		tem_obj2 = 0;
		tem_obj3 = 0;
		for v = 1:view_num
		    tem_obj1 = tem_obj1 + norm(X{v}-(B{v}+B_v{v})*H_v{v},'fro')^2 + norm(B_v{v},1) + alpha * w(v)*norm(H- H_v{v},'fro')^2;
        end
		
		for i = 1:view_num
			for j = 1:view_num
				if i ~= j
					tem_obj2 = tem_obj2 + norm(H_v{i}-H_v{j}*Z{i,j},'fro')^2 + Compute_norm21(Z{i,j}');
				end
			end
		end
		
		for v = 1:view_num
			tem = 0;
			for i = 1:view_num
				tem = tem + norm(Z{i,v},'fro')^2;
			end
			tem_obj3 = tem_obj3 + tem/(w(v)+epilision);
		end
		tem_obj3 = tem_obj3*beta/mu;
		
		obj(iter) = tem_obj1 + tem_obj2 + tem_obj3;
		if iter > 1 && abs(obj(iter) - obj(iter-1))/abs(obj(iter-1)) < 0.0001
			% break;
		end
		
	end
	
	
end