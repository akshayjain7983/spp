from configparser import ConfigParser

def readConfig(configFilename:str, configSection:str = None) -> dict:

    parser = ConfigParser()
    parser.read(configFilename)
    config = {}
    sections = parser.sections()

    for sec in sections:
        configSec = __readSection__(parser, sec)
        config.update(configSec)

    return config

def __readSection__(parser, section) -> dict:

    config = {}

    if section and parser.has_section(section):
        params = parser.items(section)
        for param in params:
            key = section+'.'+param[0]
            config[key] = param[1]

    return config