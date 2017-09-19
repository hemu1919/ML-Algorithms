function val = activation(level, input, weights)
    val=0;
    if(level~=1)
        val=sum(input(1,:).*weights(1,:));
        val=sigmf(val,[1 0]);
    end
end