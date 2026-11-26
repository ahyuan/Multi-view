function [P,Y_p,Y,mu,R,S,obj] = MDCAW(X,alpha,beta,gamma,sigma,clu_num,maxIter)
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明
%   sigma:相似度矩阵构建时的超参数
%   xi:

%期望X是列cell的形式，每一个cell表示一个视图的数据，其中每个样本列排
%view_num:视图个数
%n:样本个数
opts = optimset('Display','off');

view_num = max(size(X));
%创建一个数组记录各视图特征维度，P更新的时候有用。
dims = zeros(1,view_num);
for v = 1:view_num
	[dims(v),n] = size(X{v});
end



%初始化输出矩阵
P = cell(1,view_num);
Y_p = cell(1,view_num);
R = cell(1,view_num);
for v = 1:view_num
	P{v} = rand(dims(v),clu_num);
	R{v} = orth(rand(clu_num,clu_num));
end

Y = zeros(n,clu_num);

%%%%这个G还有用吗
mu = ones(view_num,1)./view_num;
S = zeros(n,n);
obj = zeros(maxIter, 1);

%% Initialize S_v and reinitialize S
S_v = cell(1,view_num);
% parfor v = 1:view_num
%     data_v = X{v};
%     S_temp = S;
%     for i = 1:n
%         for j = 1:n
%             S_temp(i,j) = exp(-(sum((data_v(:, i) - data_v(:, j)).^2))/sigma);
%         end
%         S_temp(i,:) = S_temp(i,:)./sum(S_temp(i,:));
%     end
%     S_v{v}=(S_temp+S_temp')./2;
% end
% clear S_temp data_v;

for v = 1:view_num
	data_v = X{v}';%%pdist要求行排列
	Dis = squareform(pdist(data_v,'squaredeuclidean'));
	Dis = -1.*Dis./sigma;
	S_temp = exp(Dis);
	rowSums = sum(S_temp,2);
	rowSum_v = repmat(rowSums,1,size(S_temp,2));
	S_temp = S_temp./rowSum_v;
	S_v{v} = (S_temp+S_temp')/2;

	L = SpectralClustering(S_v{v},clu_num);
    Temp = zeros(n,clu_num);
    for i = 1:n
        Temp(i,L(i)) = 1;
    end
	for j = 1:clu_num
		Temp(:,j) = Temp(:,j)./sqrt(sum(Temp(:,j)));
	end
	Y_p{v} = Temp;
end
clear Dis S_temp rowSum_v rowSums Temp L;

%%%initialize S and Y
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
Temp = zeros(n,clu_num);
for i = 1:n
	Temp(i,Y_pre(i)) = 1;
end
Y = Temp;
clear XX data_v Dis S_temp rowSum_v rowSums S_temp Y_pre Temp;


for iter = 1:maxIter
	%%update P
	% disp('update P...');
	for v = 1:view_num
		part_a = X{v}*X{v}'+eye(dims(v)).*alpha;
		P{v} = part_a \ (X{v}*Y_p{v});
	end
	clear part_a;
	
	%%update Y_p
	% disp('update Y_p...');
	for v = 1:view_num
		T = X{v}'*P{v};
		Z = Y;
		for m = 1:view_num
			if m~=v
				Z = Z-Y_p{m}*R{m}.*mu(m);
			end
		end
		B = T + beta.*Z*R{v}';
		LapMatrix = diag(sum(S, 1)) - S;

		Y_p{v} = gpi(LapMatrix.*gamma,B);
	end
	clear T Z LapMatrix;
	
	%%update Y
	% disp('update Y...');
	Y = zeros(n,clu_num);
	A = zeros(n,clu_num);
	for v = 1:view_num
		A = A + Y_p{v}*R{v}.*mu(v);
	end
	for i = 1:n
		[~,k] = max(A(i,:));
		Y(i,k) = 1;
	end
	clear A B;
	

	%%update mu
	% disp('update mu...');
	M = cell(1,view_num);
	Part_p = cell(1,view_num);
	Part_p_new = cell(1,view_num);
	Q = zeros(view_num,view_num);
	Q_new = zeros(view_num,view_num);
	p_temp = zeros(view_num,1);
	p_temp_new = zeros(view_num,1);
	for v = 1:view_num
		M{v} = Y_p{v}*R{v};
	end
	for v = 1:view_num
		Part_p{v} = M{v}*Y';
		Part_p_new{v} = S_v{v}*S';
	end
	p_ = zeros(view_num,1);
	LapMatrix = diag(sum(S, 1)) - S;
	for i = 1:view_num
		for j = 1:view_num
			Temp = M{i}*M{j}';
			Temp2 = S_v{i}*S_v{j}';
			Q(i,j) = trace(Temp)*beta;
			Q_new(i,j) = trace(Temp2);
		end
		p_temp(i) = -trace(Part_p{i})*beta;
		p_temp_new(i) = -trace(Part_p_new{i});
		front = norm(X{i}'*P{i}-Y_p{i},'fro')^2 + alpha.*norm(P{i},'fro')^2 + gamma.*trace(Y_p{i}'*LapMatrix*Y_p{i});
		p_(i) = 0.5*front + p_temp(i) + p_temp_new(i);
		%%因为matlab二次规划的形式中第一项有1/2，所以求解公式中第二项的2在乘1/2后消去了。
	end
	Q = Q + Q_new;
	A = -eye(view_num);
	b = zeros(view_num,1);
	Aeq = ones(1,view_num);
	beq = 1;
	mu = quadprog(Q,p_,A,b,Aeq,beq,[],[],[],opts);
	clear M Part_p Part_p_new Q Q_new p_temp p_temp_new p_ Temp Temp2 A b Aeq beq;
	
	%%update R
    for v = 1:view_num
		Z = Y;
		for m = 1:view_num
			if m~=v
				Z = Z-Y_p{m}*R{m}.*mu(m);
			end
		end
		Temp = Y_p{v}'*Z.*mu(v);
		[U,~,V] = svd(Temp);
		R{v} = U*V';
    end
	clear Z Temp U V;
    

	
	% %% update S
%     disp('update S...');%%%%%%%仔细检查ij下标的正确性
% 	H = zeros(n,n);
% 	for i = 1:view_num
% 		H = H + Y_p{i}*Y_p{i}'.*mu(i);
% 	end
% 	[V,D] = eig(H);
% 	D = sqrtm(D);
% 	F = V*D;
    
%     S_v2 = zeros(view_num,n,n);
%     for v =1:view_num
%         S_v2(v,:,:) = S_v{v};
%     end
%     F = gpuArray(F);
%     S_v2 = gpuArray(S_v2);
%     mu = gpuArray(mu);
%     gamma = gpuArray(gamma);
%     S = gpuArray(S);
%     
%     for i = 1:n
%         B_i = gpuArray.zeros(n, view_num);
%         for v = 1:view_num
%             B_i(:,v) = S_v2(v,:,i);
%         end
%         a_i = gpuArray.zeros(n, 1);
%         for p = 1:n
%             a_i(p) = norm(F(i,:)-F(p,:), 'fro')^2; %%%%%%%%%%%%将原文中ZX的部分换成H
%         end
%         part_m = B_i * mu - 0.25 * gamma * a_i; %%%%%%%%%%%%%%%%%原文中参数是alpha，本文为gamma
%         %disp('update psi...');
%         psi = gpuArray.zeros(n, 1);
%         temp = part_m - ones(n,n) * part_m ./ n + 0.5 * mean(psi) + ones(n,1)/n;   %%%%原代码好像少了1向量/n
%         for j = 1:n
%             psi(j) = max(-2*temp(j), 0);
%         end
%         temp = part_m - ones(n,n) * part_m / n + 0.5 * mean(psi) + ones(n,1)/n;   %%%%原代码好像少了1向量/n
%         for j = 1:n
%             S(i, j) = max(temp(j), 0);
%         end
%         
%         
% 
%         %S(i,:) = S(i,:)./sum(S(i,:));  %%%%%%%%%%%%归一化
%     end
% 
% 	clear H D V F B_i a_i part_m psi temp;
%     S =gather(S);
%     mu = gather(mu);
%     gamma = gather(gamma);

A = zeros(n,n);
	for v = 1:view_num	
		
%         A_temp2 = zeros(n,n);
% 		Y_p_spd = Y_p{v}';
% 		for i = 1:n
% 			for p = 1:n
% 				A_temp2(i,p) = sum((Y_p_spd(:,i)-Y_p_spd(:,p)).^2);
% 			end
%         end

%         Y_p_spd = gpuArray(Y_p{v});
        Y_p_spd = Y_p{v};
        A_temp = squareform(pdist(Y_p_spd,'squaredeuclidean'));
%         A_temp = gather(A_temp);
		A = A+ A_temp.*mu(v);
    end
    
    
	B = zeros(n,n);
	%%这里B的每一列充当原先的Bi*mu
	for v = 1:view_num
		B = B + mu(v).*S_v{v};
	end
	part_M = B - 0.25.*gamma.*A; %%?是不是这样就可以了。严格来说A好像要加转置，因为A的行对应原来的每一列。然而A是对称的
	%%原先减去均值向量，（ones(n,n) * part_m ./ n）
	Temp =part_M - mean(part_M)*eye(n) + ones(n,n)./n;
% 	Temp = gpuArray(Temp);
%     S = gpuArray(S);
    for i = 1:n
        % B_i = zeros(n, view_num);
        % for v = 1:view_num
            % B_i(:,v) = S_v{v}(:,i);
        % end
        % a_i = zeros(n, 1);
        % for p = 1:n
            % a_i(p) = norm(F(i,:)-F(p,:), 'fro')^2; %%%%%%%%%%%%将原文中ZX的部分换成H
        % end
		
		%%%本模型中的B_i*mu 不就可以直接用S_v加权和得到吗，哪用这么麻烦，所有的part-m也可以提前算
        
		% part_m = B_i * mu - 0.25 * gamma * A(i,:)'; %%%%%%%%%%%%%%%%%原文中参数是alpha，本文为gamma
        %disp('update psi...');
        psi = zeros(n, 1);
		temp = Temp(:,i) + 0.5*mean(psi);  %%直接加标量这种语法可以吗
        % temp = part_m - ones(n,n) * part_m ./ n + 0.5 * mean(psi) + ones(n,1)/n;   %%%%原代码好像少了1向量/n
%         for j = 1:n
%             psi(j) = max(-2*temp(j), 0);
%         end
        
        psi = -2*temp;
        psi(psi<0)=0;
        
		temp = Temp(:,i) + 0.5*mean(psi);  %%直接加标量这种语法可以吗
        % temp = part_m - ones(n,n) * part_m / n + 0.5 * mean(psi) + ones(n,1)/n;   %%%%原代码好像少了1向量/n
%         for j = 1:n
%             S(i, j) = max(temp(j), 0);
%         end
        
        temp(temp<0)=0;
        S(i,:) = temp;
        
        %S(i,:) = S(i,:)./sum(S(i,:));  %%%%%%%%%%%%归一化
    end
%     S = gather(S);
	clear H D V F B A A_temp Y_p_spd part_M Temp psi temp;

	%%%%%%%%%%%%%%%%%%%%%%%%%%%% update S
    % disp('update S...');
    % for i = 1:n
        % B_i = zeros(n, view_num);
        % for v = 1:view_num
            % B_i(:,v) = S_v{v}(:,i);
        % end
        % a_i = zeros(n, 1);
        % for p = 1:n
            % a_i(p) = norm(H(i,:)-H(p,:), 'fro')^2; %%%%%%%%%%%%将原文中ZX的部分换成H
        % end
        % part_m = B_i * W(:,i) - 0.25 * gamma * a_i; %%%%%%%%%%%%%%%%%原文中参数是alpha，本文为gamma
        % temp = part_m - ones(n,n) * part_m ./ n +  ones(n,1)/n;   %%%%原代码好像少了1向量/n
		% psi_mean = (1-sum(temp))/n;
        % for j = 1:n
            % S(j, i) = max((temp(j)+0.5*psi_mean), 0);
        % end
        % %S(i,:) = S(i,:)./sum(S(i,:));  %%%%%%%%%%%%归一化

    % end
    % clear B_i a_i part_m psi temp;
	
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% temp_formulation1 = 0;
    % for i =1:n
        % temp_S_j = zeros(n,1);
        % for v = 1:view_num
            % temp_S_i = temp_S_j + W(v,i)*S_v{v}(:,i);
        % end
        % temp_formulation1 = temp_formulation1 + norm(S(:,i)-temp_S_i,'fro')^2;
		% %每一列进行计算
    % end
	% temp_formulation2 = 0;
	% temp_f3_mid = zeros(n,clu_num);
	% for v = 1:view_num
		% temp_formulation2 = temp_formulation2 + norm(X{v}'*P{v}-Y_p{v},'fro')^2 + alpha*norm(P{v},'fro')^2;
		% temp_f3_mid = temp_f3_mid + Y_p{v}*R{v}.*mu(v);
	% end
	% temp_formulation3 = beta * norm(Y - temp_f3_mid)^2;
	% temp_formulation4 = gamma * trace(H'*LapMatrix * H);
    % obj(iter) = temp_formulation1 + temp_formulation2 + temp_formulation3 + temp_formulation4;
    % clear temp_formulation1 temp_S_j temp_S_i temp_formulation2 temp_f3_mid temp_formulation3 temp_formulation4;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	
	
	
    % S_F = norm(S,'fro');
    % W_F = norm(W,'fro');
    % max_s_sum = max(sum(S,1))
    % disp(iter);
	% obj(iter) = temp_formulation1;

end

end


