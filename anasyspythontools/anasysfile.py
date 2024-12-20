# -*- encoding: utf-8 -*-
#
#  anasysfile.py
#
#  Copyright 2017 Cody Schindler <cschindler@anasysinstruments.com>
#
#  This program is the property of Anasys Instruments, and may not be
#  redistributed or modified without explict permission of the author.

from xml.dom import minidom #Unfortunately required as ElementTree won't pretty format xml
import xml.etree.ElementTree as ET   #for parsing XML
import base64
import struct
import numpy as np
import re
import collections
import decimal as DC

from functools import wraps

class AnasysElement(object):


    __ignored_attrs__ = ["thumbnail"]
# class AnasysElement(collections.abc.Mapping):
    """Blank object for storing xml data"""
    def __init__(self, parent_obj=None, etree=None):
        self._parent_obj = parent_obj
        self._attributes = []   #list of dicts of tags:attributes, where applicable
        if not hasattr(self, '_iterable_write'):
            self._iterable_write = {} #just in case
        if not hasattr(self, '_special_write'):
            self._special_write = {} #just in case
        if not hasattr(self, '_special_read'):
            self._special_read = {} #just in case
        if not hasattr(self, '_skip_on_write'):
            self._skip_on_write = [] #just in case
        if etree is not None:
            self._etree_to_anasys(etree) #really just parses the hell outta this tree
        

    def __dir__(self):
        """Returns a list of user-accessible attributes"""
        vars_and_funcs = [x for x in object.__dir__(self) if x[0]!='_']
        return vars_and_funcs


    def __getitem__(self, key):
        """Class attributes can be called by subscription, e.g. Foo['bar']"""
        items = dir(self)
        # second part needed to prevent infinite recursions
        if key in items:
            return getattr(self, key)
        else:
            raise KeyError

    def __iter__(self):
        """Makes object iterable. Returns all user-accessible, non-method, attributes"""
        for obj in dir(self):
            if obj != "attrs":
                if not callable(self[obj]):
                    yield self[obj]
                
    def _get_iter_attributes(self):
        for obj in dir(self):
            if  obj != "attrs":
                if not callable(self[obj]) and not obj in self.__ignored_attrs__:
                    if isinstance(self[obj], AnasysElement):
                        for sub_obj in self[obj]._get_iter_attributes():
                            yield "{}.{}".format(obj, sub_obj[0]), sub_obj[1]
                    if isinstance(self[obj], dict):
                        for k,v in self[obj].items():
                            yield "{}.{}".format(obj, k), v
                    else:
                        yield obj, self[obj]
    @property            
    def attrs(self):
        """convenience function to get all attributes as nice dict"""
        return dict(self._get_iter_attributes())
    

    def _get_iterator(self, obj):
        """For use with _anasys_to_etree. Returns a dict to iterate over, or None"""
        #If obj is a dict, return its items
        if type(obj) == dict:
            return obj#.items()
        #If obj is derived from AnasysElement, return its user-accessible attributes that aren't in _skip_on_write
        elif isinstance(obj, AnasysElement):
            return {k: obj[k] for k in obj.__dict__.keys() if k[0] != '_' and k not in obj._skip_on_write}
        #If it's something else, return None. _anasys_to_etree will test for this condition
        else:
            return None

    def _object_to_text(self, obj):
        """Takes an object, returns it to text to append to an etree object"""
        if isinstance(obj, np.ndarray):
            return self._encode_bs64(obj)
        else:
            return str(obj)

    def _anasys_to_etree(self, obj, name="APlaceholder"):
        """Return object and all sub objects as an etree object for writing"""
        # Create new element for appending tags to
        obj_items = self._get_iterator(obj)
        #Test object list for None, indicating it's time to return some text
        if obj_items is None:
            txt = self._object_to_text(obj)
            rtn = ET.Element(name)
            rtn.text = txt
            return rtn
        #Odd case where there's no text and nothing to return
        if obj_items == {}:
            return ET.Element(name)
        #If it's made it this far, it's time to loop through obj_items
        elem = ET.Element(name)
        # pdb.set_trace()
        for k, v in obj_items.items():
            #If element was once an xml attribute, make it so again
            try: #Too lazy to deal with the fact dicts won't have this attribute
                if k in obj._attributes:
                    elem.set(k, v)
                    continue
            except: #If axz's had unique tag names I wouldn't have to do this
                pass
            #Iterable conversions
            if k in obj._iterable_write.keys():
                obj._iterable_to_etree(elem, k, v)
            #Special return values
            elif k in obj._special_write.keys():
                if callable(obj._special_write[k]):
                    obj._special_write[k](elem, k, v)
                else:
                    obj._special_write[k]
            else:
                rr = self._anasys_to_etree(v, k)
                #Create subelement k, with a value determined by recursion
                elem.append(rr)
        return elem

    def _attr_to_children(self, et_elem):
        """Convert element attributes of given etree object to child elements"""
        for attr in et_elem.items():
            ET.SubElement(et_elem, attr[0])
            et_elem.find(attr[0]).text = attr[1]

    def _etree_to_anasys(self, element, parent_obj=None):
        """Iterates through element tree object and adds atrtibutes to HeightMap Object"""
        #If element has attributes, make them children before continuing
        self._attr_to_children(element)
        # If element is a key in _special_read, set special return value
        if element.tag in self._special_read.keys():
            return self._special_read[element.tag](element)
        #If element is a key in _base_64_tags, return decoded data
        if '64' in element.tag:
            return self._decode_bs64(element.text)
        #If element has no children, return either it's text or {}
        if list(element) == []:
            if element.text:
                #Default return value for an element with text
                return element.text
            else:
                #Default return value for an empty tree leaf/XML tag
                return ""
        #If element has children, return an object with its children
        else:
            if parent_obj == None:
                #Top level case, we want to add to self, rather than blank object
                element_obj = self
            else:
                #Default case, create blank object to add attributes to
                element_obj = AnasysElement()#parent_obj=self)
            #store the etree tag name for later use
            element_obj._name = element.tag
            #Update _attributes of given element
            element_obj._attributes.extend(element.keys())
            #Loop over each child and add attributes
            for child in element:
                #Get recursion return value - either text, {} or AnasysElement() instance
                rr = element_obj._etree_to_anasys(child, element)
                #Set element_obj.child_tag = rr
                setattr(element_obj, child.tag, rr)
            #Return the object containing all children and attributes
            return element_obj

    def _check_key(self, key, _dict, copy=1):
        """Check if key is in dict. If it is, increment key until key is unique, and return"""
        if key not in _dict:
            return key
        num_list = re.findall(r'\s\((\d+)\)', key)
        if num_list != [] and key[-1] == ')':
            copy = int(num_list[-1])
        index = key.find(' ({})'.format(copy))
        if index != -1:
            key = key[:index] + ' ({})'.format(copy+1)
            return self._check_key(key, _dict, copy+1)
        else:
            key += ' ({})'.format(copy)
            return self._check_key(key, _dict, copy)

    def _decode_bs64(self, data):
        """Returns base64 data decoded in a numpy array"""
        if data is None:
            return np.ndarray(0)
        decoded_bytes = base64.b64decode(data.encode())
        fmt = 'f'*int((len(decoded_bytes)/4))
        structured_data = struct.unpack(fmt, decoded_bytes)
        decoded_array = np.array(structured_data)
        return decoded_array

    def _encode_bs64(self, np_array):
        """Returns numpy array encoded as base64 string"""
        tup = tuple(np_array.flatten())
        fmt = 'f'*np_array.size
        structured_data = struct.pack(fmt, *tup)
        encoded_string = base64.b64encode(structured_data).decode()
        return encoded_string

    def _serial_tags_to_nparray(self, parent_tag):
        """Return floats listed consecutively (e.g., background tables) as numpy array"""
        np_array = []
        for child_tag in list(parent_tag):
            np_array.append(DC.Decimal(child_tag.text))
            parent_tag.remove(child_tag)
        np_array = np.array(np_array)
        return np_array

    def _nparray_to_serial_tags(self, elem, nom, np_array):
        """Takes a numpy array returns an etree object and of consecutive <double>float</double> tags"""
        root = ET.Element(nom)
        flat = np_array.flatten()
        for x in flat:
            el = ET.SubElement(root, 'Double')
            el.text=str(x)
        elem.append(root)
        

    def write(self, filename):
        """Writes the current object to file"""
        xml = self._anasys_to_etree(self, 'Document')
        #ElementTree annoyingly only remembers namespaces that are used so next line is necessary
        xml.set("xmlns", "www.anasysinstruments.com")
        #Can't see any reason to add unused namespaces other than default, as Analysis Studio won't complain,
        #but minidom will if one is duplicated (can't easily get around this lame default behavior in etree)
        with open(filename, 'wb') as f:
            xmlstr = minidom.parseString(ET.tostring(xml)).toprettyxml(indent="  ", encoding='UTF-16')
            f.write(xmlstr)

    def _etree_to_dict(self, etree, key_tag):
        """
        Converts an ET element to a dict containing its children as AnasysElements.
        e.g.,
            <parent>
                <obj1>
                    <key>A</key>
                </obj1>
                <obj2>
                    <key>B</key>
                </obj2>
                ...
            </parent>
        becomes:
            parent = {'A': obj1, 'B': obj2, ...}
        Arguments:
            self = calling object (will be an instance of or derived from AnasysElement)
            etree = element tree object to be converted
            key_tag = object element to be used as key (e.g., Label, Name, ID, etc.)
        """
        return_dict = {}
        for child in etree:
            new_obj = AnasysElement(etree=child)
            key = new_obj[key_tag]
            key = self._check_key(key, return_dict)
            return_dict[key] = new_obj
        return return_dict

    def _etree_to_list(self, etree):
        """
        Converts an ET element to a list containing its children as AnasysElements.
        e.g.,
            <parent>
                <obj1/>
                <obj2/>
                ...
            </parent>
        becomes:
            parent = [obj1, obj2, ...]
        Arguments:
            self = calling object (will be an instance of or derived from AnasysElement)
            etree = element tree object to be converted
        """
        return_list = []
        for child in etree:
            new_obj = AnasysElement(etree=child)
            return_list.append(new_obj)
        return return_list

    def _iterable_to_etree(self, parent_elem, iterable_elem_name, iterable_obj):
        """
        Converts a named dict or list of Anasys Elements to an Element Tree
        object representation of the object

        e.g.,
            parent.var = {'ID1': obj1, 'ID2': obj2, ...} or parent.var = [obj1, obj2, ...]
            becomes:
            <parent>
                <var>
                    <obj1>...</obj1>
                    <obj2>...</obj2>
                    ...
                </var>
            </parent>
        Arguments:
        self = calling object (will be an instance of or derived from AnasysElement)
        parent_elem = the parent etree object to append to
        iterable_elem_name = the name of the dict or list variable (will become etree element name)
        iterable_obj = the dict or list itself
        """
        parent_etree = ET.SubElement(parent_elem, iterable_elem_name)
        if type(iterable_obj) == dict:
            for child in iterable_obj.values():
                new_elem = child._anasys_to_etree(child, name=child._name)
                parent_etree.append(new_elem)
        else:
            for child in iterable_obj:
                new_elem = child._anasys_to_etree(child, name=child._name)
                parent_etree.append(new_elem)
    
    def __eq__(self, other):
        for attr in set(dir(self) + dir(other)):
            v1, v2 = [getattr(obj, attr, None) for obj in [self, other]]
            if v1 is None and v2 is None: 
                return True
            if v1 is None or v2 is None:
                return False
            else:
                if isinstance(v1, np.ndarray) or isinstance(v2, np.ndarray):
                    if (v1!=v2).any():
                        return False
                if type(v1) == dict:
                    if type(v2) != dict:
                        return False
                    return np.all([np.all(v1[k] == v2[k]) for k in v1])                    
                elif not np.all(v1 == v2) and not ( callable(v1) and callable(v2)):
                    print(v1, v2)
                    return False
        return True

    
def multi_element_to_dict(datatype, element):
    ret_dict = {}
    for child in element:
        ret_dict[child.tag] = datatype(child.text)
    return ret_dict
    
def multi_element(datatype):
    def fun(element):
        return multi_element_to_dict(datatype, element)
    return fun

def simple_type(datatype):
    def fun(el):
        return datatype(el.text)
    return fun
