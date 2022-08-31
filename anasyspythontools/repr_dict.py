from .repr_utils import accordion_list
import html


class ReprDict(dict):


    def _html_or_str(self, item):
        
        try:
            return item._repr_html_()
        except AttributeError:
            return html.escape((repr(item)))

    def _repr_html_(self):
        return accordion_list([(k, self._html_or_str(v)) for k,v in self.items()])
            
