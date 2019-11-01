from enum import Enum
from enum import unique

#TODO/TDB: Pack in config/.env ?
@unique
class Messages(Enum):
    ILLEGAL_ARGUMENT_NONE_TYPE = 'illegal none-type argument'
    ILLEGAL_ARGUMENT_TYPE = 'illegal argument type'
    FILE_NOT_FOUND = 'file not found'
    PROVIDED_ARRAY_DOESNT_MATCH_DATA = ('provided array does not match data frame')
    PROVIDED_ARRAYS_DONT_MATCH_LENGTH = ('provided arrays do not match in length')
    PROVIDED_FRAME_DOESNT_MATCH_DATA = ('provided data frame does not match data frame')
    PROVIDED_MODE_DOESNT_EXIST = ('provided mode does not exist')
    NOT_IMPLEMENTED = ('requested operation not implemented')
