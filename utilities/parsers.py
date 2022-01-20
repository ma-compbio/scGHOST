import json

def parse_config(config_filepath):
    with open(config_filepath) as config_file:
        config_data = json.load(config_file)
        
        return config_data
        