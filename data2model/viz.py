import folium
import os
import webbrowser
import pandas as pd
from folium.plugins import MousePosition



WEBNAME = 'test.html'

def plot(pts):
    mid = [43.906491, -79.423148]

    m = folium.Map(location=mid, zoom_start=11)

    # Add hexagon layer


    # markers
    for pt in pts:
        folium.Marker(
            location=pt.tolist()
        ).add_to(m)

    my_js = '''
        console.log('working perfectly')
        '''
    m.get_root().script.add_child(folium.Element(my_js))
    m.save(WEBNAME)
    path_to_open = 'file:///' + os.getcwd() + '/' + WEBNAME
    webbrowser.open_new_tab(path_to_open)


# def plot():
#     mid = [43.906491, -79.423148]
#     m = folium.Map(location=mid, zoom_start=11)
#     # markers
#     m.add_child(folium.LatLngPopup())
#     m.add_child(folium.ClickForMarker(popup="Waypoint"))
#
#     # m.get_root().html.add_child(folium.JavascriptLink('../static/js/interactive_poi.js'))
#     my_js = '''
#         console.log('working perfectly')
#         '''
#     m.get_root().script.add_child(folium.Element(my_js))
#     m.save(WEBNAME)
#     path_to_open = 'file:///' + os.getcwd() + '/' + WEBNAME
#     webbrowser.open_new_tab(path_to_open)


# coordinates = pd.read_csv('../dataset/intersections.csv').to_numpy()
coordinates = pd.read_csv('../dataset/sections.csv').to_numpy()
print(coordinates)
plot(coordinates)
