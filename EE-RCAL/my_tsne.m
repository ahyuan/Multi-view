function Y = my_tsne(X, varargin)
    % t-SNE function using MATLAB's built-in tsne function
    % Parameters:
    %   X: Input data matrix (samples x features)
    %   varargin: Additional parameters for tsne function
    %
    % Example usage:
    %   Y = my_tsne(Z', 'Perplexity', 5, 'Exaggeration', 1.5);

    % Default parameters
    options = struct('Perplexity', 30, 'Exaggeration', 1, 'MaxNumIterations', 1000);

    % Parse input arguments
    if ~isempty(varargin)
        options = parse_inputs(options, varargin);
    end

    % Call MATLAB's built-in tsne function
    Y = tsne(X, 'Algorithm', 'barneshut', ...
             'Perplexity', options.Perplexity, ...
             'Exaggeration', options.Exaggeration, ...
             'MaxNumIterations', options.MaxNumIterations);
end

function options = parse_inputs(options, varargin)
    % Parse input arguments and update options structure
    valid_fields = fieldnames(options);
    for i = 1:2:length(varargin)
        if i + 1 > length(varargin)
            error('Input argument pairs must be complete.');
        end
        field = varargin{i};
        value = varargin{i+1};
        if ismember(field, valid_fields)
            options.(field) = value;
        else
            error(['Invalid parameter name: ', field]);
        end
    end
end