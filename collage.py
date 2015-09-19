from collage.fuse import Fuser
from collage.prioritize import Prioritizer

path = 'data'

fus = Fuser(path)
fus.fuse(n_run=20, dump=True)

pr = Prioritizer(path, 'seeds.tsv', 'prioritization.tsv')
pr.prioritize(n_permute=50)
