function solution = solveRationalEquation(kj, aj)
    % 输入参数:
    % kj: 一个向量，包含了所有的kj值
    % aj: 一个向量，包含了所有的aj值，需保证aj与对应的kj长度相同且aj-x不会导致分母为零
    
    % 检查输入向量长度是否一致
   if length(kj) ~= length(aj)
        error('kj和aj的长度必须一致');
    end
    
    % 定义有理方程左侧的函数
    fun = @(x) sum(kj ./ (aj - x)) - 1;
    
    initial_guess = -0.001;
    
    % 使用fsolve求解
    solution = fsolve(fun, initial_guess,optimset('Display','off'));
	
	% initial_guess = 0.001;
    
    % % 使用fsolve求解
    % solution2 = fsolve(fun, initial_guess,optimset('Display','off'));
    
	% ans1 = fun(solution1);
	% ans2 = fun(solution2);
	
	% if abs(ans1)<abs(ans2)
		% if isreal(solution1)
		% solution = solution1;
		% else
		% solution = solution2;
		% end
	% else
		% if isreal(solution2)
		% solution = solution2;
		% else 
		% solution = solution1;
        % end
	% end
	
	
    % 输出解
    % fprintf('解为: x = %f\n', solution);
end
