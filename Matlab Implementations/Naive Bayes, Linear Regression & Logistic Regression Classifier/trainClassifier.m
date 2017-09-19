function [score,models,sets] = trainClassifier(trainData,trainLabels,r_sets,d,params,options)
    data_original=sparse(trainData(:,1),trainData(:,2),trainData(:,3));
    sets.labels=trainLabels;
    sets.r_sets=r_sets;
    if(options==2)
        data_binary=data_original;
        p_labels=zeros(size(unique(trainLabels),1),1);
        docs=size(data_binary,1);
        books=size(data_binary,2);
        if(books~=d)
            for i = 1 : docs
                t=find(data_binary(i,:));
                p_labels(trainLabels(i))=p_labels(trainLabels(i))+1;
                data_binary(i,:)=data_binary(i,:)>=mean(data_binary(i,t(:)));
            end
            p_labels=p_labels/docs;
            [~,~,sets.b_ids]=selectAttribute((1:docs)',(1:books)',data_binary,trainLabels,p_labels,1,d);
        else
            sets.b_ids=(1:books)';
        end
        %books=size(sets.b_ids,1);
        data_handled=data_original;
        %for i =1:books
            %data_handled(:,sets.b_ids(i,1))=handle_missing((1:docs)',data_handled(:,sets.b_ids(i,1)),trainLabels(:,1));
        %end
        sets.data=data_handled;
        [score,models]=crossValidate(@decisionTreeTrain,@decisionTreePredict,sets,params.depth);
    else
        sets.data=data_original;
        sets.b_ids=[];
        if(options==1)
            [score,models]=crossValidate(@naiveBayesTrain,@naiveBayesPredict,sets,params.alpha);
        elseif(options==3)
            [score,models]=crossValidate(@logisticRegressionTrain,@logisticRegressionPredict,sets,params.steps);
        end
    end
end