function [nmi, clust_ent] = compute_nmi(T, H)
    
    N = length(T);
    classes = unique(T);
    clusters = unique(H);
    num_class = length(classes);
    num_clust = length(clusters);

    %% Compute number of points in each class
    for j=1:num_class
        index_class = (T(:)==classes(j));
        D(j) = sum(index_class);
    end
      
    %% Mutual information
    mi = 0;
    A = zeros(num_clust, num_class);
    avgent = 0;
    for i=1:num_clust
     
        index_clust = (H(:)==clusters(i));
        B(i) = sum(index_clust);
        for j=1:num_class
            index_class = (T(:)==classes(j));
           
            A(i,j) = sum(index_class.*index_clust);
            if (A(i,j) ~= 0)
                miarr(i,j) = A(i,j)/N * log2 (N*A(i,j)/(B(i)*D(j)));
                           
                avgent = avgent - (B(i)/N) * (A(i,j)/B(i)) * log2 (A(i,j)/B(i));
            else
                miarr(i,j) = 0;
            end
            mi = mi + miarr(i,j);
        end        
    end
    
    %% Class entropy
    class_ent = 0;
    for i=1:num_class
        class_ent = class_ent + D(i)/N * log2(N/D(i));
    end
    
  
    clust_ent = 0;
    for i=1:num_clust
        clust_ent = clust_ent + B(i)/N * log2(N/B(i));
    end
        
    mi_adjusted = mi * (1 + 0.42 / (clust_ent + class_ent)); 
    nmi = 2 * mi_adjusted / (clust_ent + class_ent);
    nmi = min(1, nmi);
end