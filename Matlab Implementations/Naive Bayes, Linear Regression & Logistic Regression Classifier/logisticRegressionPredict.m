function labels = logisticRegressionPredict(r_set, data, model,~)
    docs=size(r_set,1);
    labels=sparse(docs,1);
    for i = 1 : docs
        label=sparse(size(model,1),1);
        label=model*[1;data(r_set(i,1),(1:size(model,2)-1))'];
        [~,labels(i,1)]=max(label);
    end
end