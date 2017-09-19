function label = naiveBayesPredict(r_set, data, model,p_labels)

% FILL IN YOUR CODE AND COMMENTS HERE
    docs=size(r_set,1);
    label=zeros(docs,1);
    for i = 1 : docs
        r_id=r_set(i,1);
        w_ids= find(data(r_id,:))';
        mean_num=mean(data(r_id,:));
        data(r_id,w_ids)=data(r_id,w_ids)>=mean_num;
        product=zeros(size(p_labels,1),1);
        for k = 1 : size(p_labels,1)
            product(k,1)=p_labels(k,1);
            for j = 1 : size(w_ids,1)
                temp=w_ids(j,1);
                if(temp>size(model,1)-1)
                    temp=size(model,1);
                end
                product(k,1)=product(k,1)+model(temp,data(r_id,temp)+1,k);
            end
        end
        [~,label(i,1)]=max(product);
    end
end
