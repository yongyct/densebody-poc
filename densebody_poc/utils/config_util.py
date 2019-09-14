import json

def get_json_conf(filename):
    '''
    Parses json information in the filename, and return it as a dictionary
    '''
    with open(filename) as json_file:
        conf = json.load(json_file)
        
    return conf
