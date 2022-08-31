from .repr_utils import accordion_list



class ReprDict(dict):


    def _html_or_str(self, item):
        try:
            return item._repr_html_()
        except:
            return str(item)

    def _repr_html_(self):
        return accordion_list([(k, self._html_or_str(v)) for k,v in self.items()])
            
