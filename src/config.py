
import os

# required python version
required_python_version_major = 2
required_python_version_minor = 7

# file naming
temp_dir=""
unnamed_temp_dir=""
file_basename=""



# output file formats
output_file_column_delimiter="\t"
output_file_category_delimiter="|"

def get_halla_base_directory():
    """ 
    Return the location of the halla base directory
    """
    
    config_file_location=os.path.dirname(os.path.realpath(__file__))
    
    # The halla base directory is parent directory of the config file location
    halla_base_directory=os.path.abspath(os.path.join(config_file_location,os.pardir))
    
    return halla_base_directory