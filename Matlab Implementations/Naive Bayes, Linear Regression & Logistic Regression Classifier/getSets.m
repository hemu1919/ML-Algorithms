function sets = getSets(trainData,folds)
    docs=size(unique(trainData(:,1)),1);
    max_size=round(docs/folds);
    visited=sparse(docs,1);
    sets=sparse(folds,max_size);
    hold_set=sparse(max_size,1);
    for i = 1 : folds
        while(size(find(hold_set),1)<max_size && i<folds)
            r_id=round(rand()*docs);
            flag=size(find(visited==r_id),1);
            if(flag~=0)
                continue;
            end
            visited(size(find(visited),1)+1)=r_id;
            hold_set(size(find(hold_set),1)+1)=r_id;
        end
        if(i==folds)
            hold_set=setdiff(1:docs,visited)';
        end
        sets(i,1:size(hold_set,1))=hold_set';
        hold_set(:,:)=0;
    end
end