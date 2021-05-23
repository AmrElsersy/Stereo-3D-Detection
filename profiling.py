import cProfile
import pstats

p = pstats.Stats('output.txt')
p.strip_dirs().sort_stats("cumulative").print_stats()
