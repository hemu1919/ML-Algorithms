function [model,p_labels] = decisionTreeTrain(r_set, b_ids, trainData, trainLabels, params)

% FILL IN YOUR CODE AND COMMENTS HEREdocs=size(trainData,2);
    docs=size(r_set,1);
    p_labels=zeros(size(unique(trainLabels),1),1);
    model=struct('node',0,'node0',0,'node1',0);
    for i = 1 : docs
        p_labels(trainLabels(r_set(i,1)))=p_labels(trainLabels(r_set(i,1)))+1;
        w_ids=b_ids(find(trainData(r_set(i,1),b_ids)),1);
        mean_num=mean(trainData(r_set(i,1),b_ids));
        trainData(r_set(i,1),w_ids)=trainData(r_set(i,1),w_ids)>=mean_num;
    end
    p_labels=p_labels/docs;
    model=buildModel(r_set,b_ids,trainData,trainLabels,p_labels,model,1,params);
    p_labels=0;
end
function sets = subset(r_set, b_ids, data, b_id)
    sets.b_ids=setdiff(b_ids,b_id);
    sets.set1.r_set=r_set(find(data(r_set(:,1),b_id)),1);
    sets.set0.r_set=setdiff(r_set,sets.set1.r_set);
end
function model = buildModel(r_set, b_ids, data, labels, p_labels, model, n_id, depth)
    y=-1;
    if(mod(n_id,2)==0)
        x=n_id;
    else
        x=n_id-1;
    end
    while(x>0)
        x=floor(x/2);
        y=y+1;
    end
    if(y==depth)
        count=sparse(size(unique(labels),1),1);
        for i = 1 : size(r_set,1)
            count(labels(r_set(i,1),1))=count(labels(r_set(i,1),1))+1;
        end
        [~,label]=max(count);
        if((ceil(n_id/2)-floor(n_id/2))==0)
            model(floor(n_id/2)).node0=label;
        elseif((ceil(n_id/2)-floor(n_id/2))==1)
            model(floor(n_id/2)).node1=label;
        end
        return;
    end
    [gain,a_id,set]=selectAttribute(r_set,b_ids,data,labels,p_labels,0,-1);
    if(gain==-1)
       if((ceil(n_id/2)-floor(n_id/2))==0)
            model(floor(n_id/2)).node0=a_id;
       elseif((ceil(n_id/2)-floor(n_id/2))==1)
            model(floor(n_id/2)).node1=a_id;
       end
       return;
    end
    model(n_id).node=b_ids(a_id);
    model(n_id).node0=0;
    model(n_id).node1=0;
    sets=subset(r_set,b_ids,data,b_ids(a_id));
    model=buildModel(sets.set0.r_set,sets.b_ids,data,labels,set.zero,model,2*n_id,depth);
    model=buildModel(sets.set1.r_set,sets.b_ids,data,labels,set.one,model,(2*n_id)+1,depth);
end
