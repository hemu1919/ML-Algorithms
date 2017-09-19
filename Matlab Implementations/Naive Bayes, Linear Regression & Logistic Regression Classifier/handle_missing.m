function attr = handle_missing(r_set,attr,labels)
    r_ids=r_set(find(attr(r_set,1)),1);
    r_ids=setdiff(r_set,r_ids)';
    label=unique(labels(r_set,1));
    for i = 1 : size(label,1)
        lab=label(i,1);
        r_l_ids=r_set(find(labels(r_set,1)==lab),1);
        temp=unique(attr(r_l_ids,1));
        max_count=0;value=0;
        for j = 1 : size(temp,1)
            if(sum(attr(r_l_ids,1)==temp(j,1))>max_count)
                value=temp(j,1);
            end
        end
        attr(intersect(r_l_ids,r_ids)')=value;
    end
end