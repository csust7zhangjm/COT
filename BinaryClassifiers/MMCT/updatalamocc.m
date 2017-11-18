function new_lRate = updatalamocc(new_lRate,clesszero)

for i=1:clesszero
    new_lRate = new_lRate+(1.0-new_lRate)*(clesszero/(1+clesszero));
end

