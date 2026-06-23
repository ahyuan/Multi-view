function [P,H,Z,S,obj,iter] = MSCLSP(X,alpha,beta,gama,lamda,epilsion,maxIter,K)
% alpha beta gama lamda ³¬²ÎÊı
% output 
% P the project matrix
% H latent representation
% Z self-representation
% S similarity matrix
V = size(X,2);    
N = size(X{1},2); 
for i=1:V
    D{i}=size(X{i},1);             
end
SD=0;
M=[];
for i=1:V
    SD = SD+D{i};                    
    M = [M;X{i}];                  
end
% initial
% H = getPCA(M',K);
P = zeros(SD,K);
H = rand(K,N);   
% Z = zeros(N,N);
Z = rand(N,N);
S = zeros(N,N);
obj = [];
allacc = [];
iter=1;err=1;
while (err>0.00001 && iter<=maxIter)

   % update S
   for i=1:N
       for j=1:N    
             temp(i,j) = exp(-1*gama*(norm(H(:,i)-H(:,j),2)^2)/(lamda+1e-11));   
       end
       theta(i) = lamda *(1-log(sum(temp(i,:))));     
   end
   for i=1:N
       for j=1:N
             S(i,j) = exp((theta(i) - gama*norm(H(:,i)-H(:,j),2)^2)/(lamda+1e-11));
       end
       S(i,:) = S(i,:)./sum(S(i,:));
   end
   S=(S+S')./2;
   % update H
   LapMatrix = diag(sum(S,2))-S;  %L
   A = eye(N)+alpha *(eye(N) - Z - Z' + Z*Z')+gama*LapMatrix;
   C = P'* M;
   H = C/A;
   %% fixed H,P,S update Z
   
   iter_Z=1;
   while iter_Z<maxIter
       Q = diag(0.5./sqrt(sum(Z.*Z,2)+epilsion));
       Z = real(alpha * inv(alpha* H'* H+beta*Q)*H'*H);
       iter_Z = iter_Z+1;
   end
%    Q = diag(0.5./sqrt(sum(Z.*Z,2)+epilsion));
%    Z = real(alpha * inv(alpha* H'* H+beta*Q)*H'*H);
   
   %% fixed Z,H,S update P
   temp_M = M*H';
   [svd_U,~,svd_V] = svd(temp_M,'econ');
   P = svd_U*svd_V';
   
 
   temp_formulation1 = norm(M-P*H,'fro')^2;
 
   temp_formulation2 = alpha*norm(H-H*Z,'fro')^2;

   tmp1 = zeros(N,1);
   for i=1:N
       tmp1(i)=norm(Z(i,:),2);
   end
   temp_formulation3=beta*norm(tmp1,1);

   LapMatrix = diag(sum(S,2))-S;
   temp_formulation4 = gama*trace(H*LapMatrix*H');
   
   tmp2 = zeros(N,1);
   for i=1:N
       tmp2(i) = sum(S(:,i).*log(S(:,i)));
   end
   temp_formulation5=lamda*sum(tmp2);
   obj(iter) = temp_formulation1+temp_formulation2+temp_formulation3+temp_formulation4+temp_formulation5;%æ€»å??
   if iter>2
        err = abs(obj(iter-1)-obj(iter));
   end

   iter = iter+1;
   

end

