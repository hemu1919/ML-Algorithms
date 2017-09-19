function label = decisionTreePredict(r_set, b_ids, data, model,~)

% FILL IN YOUR CODE AND COMMENTS HERE
    docs=size(r_set,1);
    label=sparse(docs,1);
    for i=1:docs
        j=1;
        mean_num=mean(data(r_set(i,1),b_ids));
        data(r_set(i,1),b_ids)=data(r_set(i,1),b_ids)>=mean_num;
        while(j<=size(model,2))
            t=data(r_set(i,1),model(1,j).node);
            if(t==0)
                if(model(1,j).node0==0)
                    j=2*j;
                    continue;
                end
                label(i,1)=model(1,j).node0;
                break;
            elseif(t==1)
                if(model(1,j).node1==0)
                    j=(2*j)+1;
                    continue;
                end
                label(i,1)=model(1,j).node1;
                break;
            end
        end
    end
end
