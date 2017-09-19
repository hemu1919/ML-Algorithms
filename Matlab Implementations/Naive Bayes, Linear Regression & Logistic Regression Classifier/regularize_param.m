function regularize_param(trainData,trainLabels,r_sets,params_set,options)
    data_original=sparse(trainData(:,1),trainData(:,2),trainData(:,3));
    sets.labels=trainLabels;
    sets.r_sets=r_sets;
    accuracy=sparse(1,size(params_set,2));
    avg_accuracy=sparse(1,size(params_set,2));
    if(options==2)
        data_binary=data_original;
        p_labels=zeros(size(unique(trainLabels),1),1);
        for i = 1 : docs
            t=find(data(i,:));
            p_labels(trainLabels(i))=p_labels(trainLabels(i))+1;
            data_binary(i,:)=data(i,:)>=mean(data_binary(i,t(:)));
        end
        p_labels=p_labels/docs;
        [~,~,sets.b_ids]=selectAttribute((1:docs)',(1:books)',data_binary,trainLabels,p_labels,1,d);
        books=size(sets.b_ids,1);
        data_handled=data_original;
        for i =1:books
             data_handled(:,sets.b_ids(i,1))=handle_missing((1:docs)',data_handled(:,sets.b_ids(i,1)),trainLabels(:,1));
        end
        sets.data=data_handled;
    else
        sets.data=data_original;
        sets.b_ids=[];
    end
    for j = 1 : size(params_set,2)
        if(options==2)
            [score,~]=crossValidate(@decisionTreeTrain,@decisionTreePredict,sets,params_set(1,j));
        else
            if(options==1)
                [score,~]=crossValidate(@naiveBayesTrain,@naiveBayesPredict,sets,params_set(1,j));
            elseif(options==3)
                [score,~]=crossValidate(@logisticRegressionTrain,@logisticRegressionPredict,sets,params_set(1,j));
            end
        end
        avg_accuracy(1,j)=score.Avg_Accuracy;
        accuracy(1,j)=score.Accuracy;
    end
    figure,plot(params_set,accuracy),xlabel('Parameter Values'),ylabel('Accuracy'),grid on,axis equal
    figure,plot(params_set,avg_accuracy),xlabel('Parameter Values'),ylabel('Average Accuracy'),grid on,axis equal
end