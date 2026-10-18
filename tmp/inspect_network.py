import pypsa
n=pypsa.Network(r"results/phase_06f/F06F_P45/network_solved_2022.nc")
print(n.generators.loc[n.generators.carrier.isin(['interconnection_colombia','load_shedding']),['bus','carrier','p_nom','marginal_cost']].to_string())
print(n.links[['bus0','bus1','carrier','p_nom']].to_string())
print('loads energy',n.loads_t.p_set.sum().sum()/1000)
print('gen sums')
print(n.generators_t.p.sum()/1000)
print('link p0/p1 sums',n.links_t.p0.sum()/1000,n.links_t.p1.sum()/1000)
