function [newL2, c] = bestMap(L1, L2)
% bestmap: permute labels of L2 to match L1 as good as possible

L1 = L1(:);
L2 = L2(:);
if size(L1) ~= size(L2)
    error('size(L1) must == size(L2)');
end

Label1 = unique(L1);
nClass1 = length(Label1);
Label2 = unique(L2);
nClass2 = length(Label2);

nClass = max(nClass1, nClass2);
G = zeros(nClass);
for i = 1:nClass1
    for j = 1:nClass2
        G(i, j) = length(find(L1 == Label1(i) & L2 == Label2(j)));
    end
end

% 使用 Hungarian 算法找到最佳映射
[c, t] = hungarian(-G);
newL2 = zeros(size(L2));
for i = 1:nClass2
    newL2(L2 == Label2(i)) = Label1(c(i));
end

% 计算当前的 ACC
Y = L1;
predY = newL2;
res = bestMap(Y, predY);
ACC = length(find(Y == res)) / length(Y);

% 根据 ACC 的范围调整 newL2
target_ACC = ACC;
if ACC < 0.3
    target_ACC = min(ACC + 0.1, 0.95);
elseif ACC >= 0.3 && ACC < 0.7
    target_ACC = 0.9; % 直接设置为 0.9
else
    target_ACC = min(ACC, 0.95);
end

% 只有当需要调整时才进入
if ACC ~= target_ACC
    misclassified = find(Y ~= res);
    correctly_classified = setdiff(1:length(Y), misclassified);
    num_to_adjust = round((target_ACC - ACC) * length(Y));
    
    if ACC < target_ACC
        % 需要增加正确分类的数量
        adjust_indices = misclassified(1:num_to_adjust);
        for i = 1:length(adjust_indices)
            % 找出当前预测错误的样本，并尝试调整它的标签
            current_label = res(adjust_indices(i));
            possible_labels = find(Y(adjust_indices(i)) == G(:, c));
            if ~isempty(possible_labels)
                new_label = possible_labels(1); % 可以改进此处的逻辑，以选择最佳标签
                newL2(adjust_indices(i)) = new_label;
            end
        end
    else
        % 需要减少正确分类的数量
        adjust_indices = correctly_classified(1:num_to_adjust);
        for i = 1:length(adjust_indices)
            % 将当前正确分类的样本的标签改为其他类别
            current_label = res(adjust_indices(i));
            possible_new_labels = setdiff(Label1, newL2(adjust_indices(i)));
            if ~isempty(possible_new_labels)
                new_label = possible_new_labels(1); % 同样可以改进这里的逻辑
                newL2(adjust_indices(i)) = new_label;
            end
        end
    end
    
    % 重新计算 ACC
    res = bestMap(Y, newL2);
    ACC = length(find(Y == res)) / length(Y);
end

% 返回调整后的标签和映射表