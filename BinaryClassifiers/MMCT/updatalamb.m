function new_lRateb = updatalamb(new_lRateb,clesszero)

    new_lRateb = 0.9*new_lRateb-0.1*(clesszero/(1+clesszero));
end

