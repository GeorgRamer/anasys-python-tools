
try:
    import ipywidgets as widgets
except ModuleNotFoundError:
    widget_available = False
else:
    widget_available = True
    

def repr_tag_dict_widget(tag_dict):
    items = []
    for k,v in tag_dict.items():
        if k == "SampleBase64" or k=="Tags":
            continue
        items.append(widgets.Label(str(k)))
        items.append(widgets.Label(str(v)))
    if "Tags" in tag_dict:
        items.append(widgets.Label("Tags"))
        items.append(widgets.Label(""))
        for k,v in tag_dict["Tags"].items():
            items.append(widgets.Label(str(k)))
            items.append(widgets.Label(str(v)))
    return widgets.GridBox(items, layout=widgets.Layout(grid_template_columns="repeat(2,100px)"))
    
    
def image_and_tags_widget(image, tag_dict, height="100px"):
    children = [widgets.Image(value=image, format="png"),repr_tag_dict_widget(tag_dict)]
    tab =  widgets.Tab()
    tab.children = children
    tab.titles = ["Image", "Metadata"]
    return tab


def repr_tag_dict_html(tag_dict):
    row_str = "<tr><td>{}</td><td>{}</td></tr>"
    outside ="<table><tbody>{}</tbody></table>"
    items = []
    for k,v in tag_dict.items():
        if "SampleBase64" in k or k=="Tags" or ".signal" in k or ".wn" in k:
            continue
        items.append(row_str.format(k,v))
    if "Tags" in tag_dict:
        items.append(row_str.format("Tags", ""))
        for k,v in tag_dict["Tags"].items():
            items.append(row_str.format(k,v))
    return outside.format("\n".join(items))
    
    
    
def accordion_list(list_of_elements):

    element = """
    <div class="accordion">{title}</div>
<div class="panel">
  {content}
</div>
    """
    header = """<style>
            
            /* Style the buttons that are used to open and close the accordion panel */
        .accordion {
          background-color: #eee;
          color: #444;
          cursor: pointer;
          padding: 18px;
          width: 100%;
          text-align: left;
          border: none;
          outline: none;
          transition: 0.4s;
        }

        /* Add a background color to the button if it is clicked on (add the .active class with JS), and when you move the mouse over it (hover) */
        .active, .accordion:hover {
          background-color: #ccc;
        }
        .panel {
          padding: 0 18px;
          background-color: white;
          max-height: 0;
          overflow: hidden;
          transition: max-height 0.2s ease-out;
        }
        </style>

        <script>
        var acc = document.getElementsByClassName("accordion");
        var i;

        for (i = 0; i < acc.length; i++) {
          acc[i].addEventListener("click", function() {
            this.classList.toggle("active");
            var panel = this.nextElementSibling;
            if (panel.style.maxHeight) {
              panel.style.maxHeight = null;
            } else {
              panel.style.maxHeight = panel.scrollHeight + "px";
            }
          });
        }
        </script>"""
        
    content = [element.format(title=title, content=content) for title, content in list_of_elements]
    return header + "\n".join(content)
    
    
    
 
    
        
