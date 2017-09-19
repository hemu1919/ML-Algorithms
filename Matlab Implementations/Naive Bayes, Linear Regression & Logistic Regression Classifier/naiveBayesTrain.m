function [model,p_labels] = naiveBayesTrain(r_set, trainData, trainLabels, alpha)
    
    % FILL IN YOUR CODE AND COMMENTS HERE
    docs=size(r_set,1);
    books=size(trainData,2)+1;
    model=zeros(books,2,size(unique(trainLabels),1));
    p_labels=zeros(size(unique(trainLabels),1),1);
    for i = 1 : docs
        r_id=r_set(i,1);
        p_labels(trainLabels(r_id),1)=p_labels(trainLabels(r_id),1)+1;
        w_ids= find(trainData(r_id,:))';
        mean_num=mean(trainData(r_id,:));
        trainData(r_id,w_ids)=trainData(r_id,w_ids)>=mean_num;
        for j = 1 : size(w_ids,1)
            model(w_ids(j,1),trainData(r_id,w_ids(j,1))+1,trainLabels(r_id))=model(w_ids(j,1),trainData(r_id,w_ids(j,1))+1,trainLabels(r_id))+1;
        end
    end
    for i = 1 : size(p_labels,1)
        model(:,:,i)=(model(:,:,i)+alpha)/(p_labels(i)+(2*alpha));
    end
    p_labels=log(p_labels/docs);
    model=log(model);
end
