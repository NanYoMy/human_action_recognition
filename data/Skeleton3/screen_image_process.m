a1=1;
a2=18;
s1=1;
s2=10;
e1=1;
e2=3;

for a=a1:a2
    for s=s1:s2
        for e=e1:e2
            file=sprintf('./screen/a%02i_s%02i_e%02i_screen.txt',a,s,e);
            toRGB(file);
        end
    end
end



