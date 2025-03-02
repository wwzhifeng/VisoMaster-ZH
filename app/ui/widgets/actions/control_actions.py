from typing import TYPE_CHECKING
import torch
import qdarkstyle
from PySide6 import QtWidgets 
import qdarktheme

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow
from app.ui.widgets.actions import common_actions as common_widget_actions

#'''
#    Define functions here that has to be executed when value of a control widget (In the settings tab) is changed.
#    The first two parameters should be the MainWindow object and the new value of the control 
#'''

def change_execution_provider(main_window: 'MainWindow', new_provider):
    main_window.video_processor.stop_processing()
    main_window.models_processor.switch_providers_priority(new_provider)
    main_window.models_processor.clear_gpu_memory()
    common_widget_actions.update_gpu_memory_progressbar(main_window)

def change_threads_number(main_window: 'MainWindow', new_threads_number):
    main_window.video_processor.set_number_of_threads(new_threads_number)
    torch.cuda.empty_cache()
    common_widget_actions.update_gpu_memory_progressbar(main_window)


def change_theme(main_window: 'MainWindow', new_theme):

    def get_style_data(filename, theme='dark', custom_colors=None):
        custom_colors = custom_colors or {"primary": "#4facc9"}
        with open(f"app/ui/styles/{filename}", "r") as f: # pylint: disable=unspecified-encoding
            _style = f.read()
            _style = qdarktheme.load_stylesheet(theme=theme, custom_colors=custom_colors)+'\n'+_style
        return _style
    app = QtWidgets.QApplication.instance()

    _style = ''
    if new_theme == "Dark":
        _style = get_style_data('dark_styles.qss', 'dark',)

    elif new_theme == "Light":
        _style = get_style_data('light_styles.qss', 'light',)

    elif new_theme == "Dark-Blue":
        _style = get_style_data('dark_styles.qss', 'dark',) + qdarkstyle.load_stylesheet() # Applica lo stile dark-blue 

    app.setStyleSheet(_style)

    main_window.update()  # Aggiorna la finestra principale

def set_video_playback_fps(main_window: 'MainWindow', set_video_fps=False):
    # print("Called set_video_playback_fps()")
    if set_video_fps and main_window.video_processor.media_capture:
        main_window.parameter_widgets['VideoPlaybackCustomFpsSlider'].set_value(main_window.video_processor.fps)

def toggle_virtualcam(main_window: 'MainWindow', toggle_value=False):
    video_processor = main_window.video_processor
    if toggle_value:
        video_processor.enable_virtualcam()
    else:
        video_processor.disable_virtualcam()

def enable_virtualcam(main_window: 'MainWindow', backend):
    print('backend', backend)
    main_window.video_processor.enable_virtualcam(backend=backend)