"""
This file opens a web map of Richmond Hill.

You can click on the map, and it will display the coordinates of that clicked location.

This file is used as a first step to select the intersections of the Richmond Hill area.

After you completed with the selection, right click on the map and select `inspect element`,
which will show the html code.

Copy the code and paste it in ../dataset/intersection_selection_result.txt
 -> this step has to be manual as there is will take too much unnecessary effort to
    create a web app solely for this purpose.

"""
import webbrowser
import os

WEBNAME = '../dataset/intersection_selector.html'

path_to_open = 'file:///' + os.getcwd() + '/' + WEBNAME
webbrowser.open_new_tab(path_to_open)