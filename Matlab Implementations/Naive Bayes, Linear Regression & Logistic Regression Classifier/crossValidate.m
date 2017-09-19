function [score, models] = crossValidate(trainer, predictor, sets, params)
    data=sets.data;
    labels=sets.labels;
    b_ids=sets.b_ids;
    models.b_ids=b_ids;
    folds=size(sets.r_sets,1);
    docs=size(data,1);
    accur=sparse(folds,1);
    max_accur=0;
    for i = 1 : folds
        hold_set=sets.r_sets(i,:)';
        if(i==folds)
            temp=size(data,1)-(folds-1)*size(hold_set,1);
            hold_set=hold_set(1:temp,1);
        end
        if(size(b_ids,1)==0)
            [model,p_labels]=trainer(setdiff(1:docs,hold_set)',data,sets.labels,params);
            label=predictor(hold_set,data,model,p_labels);
        else
            [model,p_labels]=trainer(setdiff(1:docs,hold_set)',b_ids,data,sets.labels,params);
            label=predictor(hold_set,b_ids,data,model,p_labels);
        end
        accuracy=100*size(find(labels(hold_set)==label),1)/size(hold_set,1);
        accur(i,1)=accuracy;
        if((i>1 && max_accur<accuracy) || i==1)
            max_accur=accuracy;
            models.Model=model;
            score.Accuracy=accuracy;
            models.Labels=p_labels;
        end
    end
    score.Avg_Accuracy=mean(accur);
end