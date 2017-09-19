function model = buildmodel(model, iter, train_data, train_label)
    y=sparse(1,size(unique(train_label),1));
    s=[];
    for iter_count = 1 : iter
        for i = 1 : size(train_data,1)
            a=struct('level1',[1 train_data(i,:)]);
            y(train_label(i)+1)=1;
            for j = 2 : model.levels
                level=getfield(model,strcat('level',int2str(j)));
                a1=[];
                for k = 1 : level.count
                    a1(end+1)=activation(j,getfield(a,strcat('level',int2str(j-1))),getfield(level.weights,strcat('node',int2str(k))));
                end
                a=setfield(a,strcat('level',int2str(j)),[1 a1]);
            end
            %save('a.mat','a');
            temp=getfield(a,strcat('level',int2str(model.levels)));
            t=sum((y-temp(1,2:end)).^2)/2;
            derivatives=struct('entry1',-(y-temp(1,2:end)));
            for j = model.levels : -1 : 2
                level=getfield(model,strcat('level',int2str(j)));
                weights=level.weights;
                e=getfield(derivatives,strcat('entry',int2str(model.levels-j+1)));
                e1=getfield(a,strcat('level',int2str(j)));
                e1=e1(1,2:end);
                e2=getfield(a,strcat('level',int2str(j-1)));
                interval1=0;
                for k = 1 : level.count
                    node=getfield(weights,strcat('node',int2str(k)));
                    first=e(k);
                    y1=e1(k);
                    interval=first*y1*(1-y1);
                    interval1=interval1+interval*node(1,2:end);
                    change=-(level.step*interval*e2);
                    level.weights=setfield(level.weights,strcat('node',int2str(k)),node+change);
                end
                derivatives=setfield(derivatives,strcat('entry',int2str(model.levels-j+2)),interval1);
                model=setfield(model,strcat('level',int2str(j)),level);
            end
            y(:)=0;
        end
        accuracy=sum(nnpredict(model,train_data)==train_label)/size(train_data,1);
        s(end+1,:)=[iter_count,t,accuracy];
        if(t <= 0.001)
            break
        end
    end
    %save('temp.mat','s');
end