import sys
import json
from visualize import *

def main():
    config, data_file = sys.argv[1], sys.argv[2]
    config = json.load(open(config, 'r'))
    print(config.keys())
    data_file = json.load(open(data_file, 'r'))
    success, msg = render_graph(config, data_file, './')
    print(success, msg)

main()