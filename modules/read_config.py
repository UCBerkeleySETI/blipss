'''
This file is an adapted version of https://github.com/federatedcloud/FRB_pipeline/blob/master/Pipeline/readconfig.py.

BSD 3-Clause License

Copyright (c) 2018, Federated Cloud
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''
import sys
import configparser
from collections import OrderedDict
##########################################################
true_values = ['True', 'true', 'TRUE', 'T', 't']
false_values = ['False', 'false', 'FALSE', 'F', 'f']

config = configparser.SafeConfigParser(dict_type=OrderedDict, allow_no_value=True)
config.optionxform = str
##########################################################
def print_config(config):
    for x in config.sections():
        print("[%s]"%(x))
        for(x, value) in config.items(x):
            print("%s=%s"%(x,value))
        print()

def read_config(filename, dictionary={}):
    config.read(filename)
    # Put information in the dirctionary
    for x in config.sections():
        for(x, value) in config.items(x):
            dictionary[x] = remove_spaces(remove_comments(value))
    dictionary['methods'] = config.sections()
    dictionary = convert_values(dictionary)
    return dictionary

def remove_comments(value):
    return value.split(";")[0]

def remove_spaces(value):
    temp = value.strip()
    return temp

# convert dictionary string values to float or int if they are numbers
def convert_values(d):
    for x in d:
        if isinstance(d[x], str):
            if is_bool(d[x]):
                d[x] = to_bool(d[x])
            elif is_int(d[x]):
                d[x] = int(d[x])
            elif is_float(d[x]):
                d[x] = float(d[x])
            elif '[' and ']' in d[x]:
                d[x] = eval(d[x])
            else:
                continue
    return d

# check if input can be converted to bool, return bool (of if it can)
def is_bool(input):
    if ((input in true_values) or (input in false_values)):
        return True
    else:
        return False

def to_bool(input):
    if input in true_values:
        return True
    else:
        return False

# check if input can be converted to float, return bool
def is_float(input):
    try:
        n = float(input)
    except ValueError:
        return False
    return True

# check if input can be converted to int, return bool
def is_int(input):
    try:
        n = int(input)
    except ValueError:
        return False
    return True
##########################################################
