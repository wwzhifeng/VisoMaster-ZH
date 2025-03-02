from typing import TYPE_CHECKING
from functools import partial

from PySide6 import QtWidgets, QtGui, QtCore
from app.ui.widgets.actions import list_view_actions
from app.ui.widgets import ui_workers
import app.helpers.miscellaneous as misc_helpers

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow

class GraphicsViewEventFilter(QtCore.QObject):
    def __init__(self, main_window: 'MainWindow', parent=None):
        super().__init__(parent)
        self.main_window = main_window

    def eventFilter(self, graphics_object: QtWidgets.QGraphicsView, event):
        if event.type() == QtCore.QEvent.Type.MouseButtonPress:
            if event.button() == QtCore.Qt.MouseButton.LeftButton:
                self.main_window.buttonMediaPlay.click()
                # You can emit a signal or call another function here
                return True  # Mark the event as handled
        return False  # Pass the event to the original handler
    
class videoSeekSliderLineEditEventFilter(QtCore.QObject):
    def __init__(self, main_window: 'MainWindow', parent=None):
        super().__init__(parent)
        self.main_window = main_window
    
    def eventFilter(self, line_edit: QtWidgets.QLineEdit, event):
        if event.type() == QtCore.QEvent.KeyPress:
            # Check if the pressed key is Enter/Return
            if event.key() in (QtCore.Qt.Key_Enter, QtCore.Qt.Key_Return):            
                new_value = line_edit.text()
                # Reset the line edit value to the slider value if the user input an empty text
                if new_value=='':
                    new_value = str(self.main_window.videoSeekSlider.value())
                else:
                    new_value = int(new_value)
                    max_frame_number = self.main_window.video_processor.max_frame_number
                    # If the value entered by user if greater than the max no of frames in the video, set the new value to the max_frame_number
                    if new_value > max_frame_number:
                        new_value = max_frame_number
                # Update values of line edit and slider
                line_edit.setText(str(new_value))
                self.main_window.videoSeekSlider.setValue(new_value)
                self.main_window.video_processor.process_current_frame()  # Process the current frame

                return True
        return False
    
class VideoSeekSliderEventFilter(QtCore.QObject):
    def __init__(self, main_window: 'MainWindow', parent=None):
        super().__init__(parent)
        self.main_window = main_window

    def eventFilter(self, slider, event):
        if event.type() == QtCore.QEvent.Type.KeyPress:
            if event.key() in {QtCore.Qt.Key_Left, QtCore.Qt.Key_Right}:
                # Allow default slider movement
                result = super().eventFilter(slider, event)
                
                # After the slider moves, call the custom processing function
                QtCore.QTimer.singleShot(0, self.main_window.video_processor.process_current_frame)
                
                return result  # Return the result of the default handling
        elif event.type() == QtCore.QEvent.Type.Wheel:
            # Allow default slider movement
            result = super().eventFilter(slider, event)
            
            # After the slider moves, call the custom processing function
            QtCore.QTimer.singleShot(0, self.main_window.video_processor.process_current_frame)
            return result

        # For other events, use the default behavior
        return super().eventFilter(slider, event)
    
class ListWidgetEventFilter(QtCore.QObject):
    def __init__(self, main_window: 'MainWindow', parent=None):
        super().__init__(parent)
        self.main_window = main_window

    def eventFilter(self, list_widget: QtWidgets.QListWidget, event: QtCore.QEvent|QtGui.QDropEvent|QtGui.QMouseEvent):
        
        if list_widget == self.main_window.targetVideosList or list_widget == self.main_window.targetVideosList.viewport():

            if event.type() == QtCore.QEvent.Type.MouseButtonPress:
                if event.button() == QtCore.Qt.MouseButton.LeftButton and not self.main_window.target_videos:
                    list_view_actions.select_target_medias(self.main_window, 'folder')

            elif event.type() == QtCore.QEvent.Type.DragEnter:
                # Accept drag events with URLs
                if event.mimeData().hasUrls():

                    urls = event.mimeData().urls()
                    print("Drag: URLS", [url.toLocalFile() for url in urls])
                    event.acceptProposedAction()
                    return True
            # Handle the drop event
            elif event.type() == QtCore.QEvent.Type.Drop:

                if event.mimeData().hasUrls():
                    # Extract file paths
                    file_paths = []
                    for url in event.mimeData().urls():
                        url = url.toLocalFile()
                        if misc_helpers.is_image_file(url) or misc_helpers.is_video_file(url):
                            file_paths.append(url)
                        else:
                            print(f'{url} is not an Video or Image file')                    
                    # print("Drop: URLS", [url.toLocalFile() for url in urls])
                    if file_paths:
                        self.main_window.video_loader_worker = ui_workers.TargetMediaLoaderWorker(main_window=self.main_window, folder_name=False, files_list=file_paths)
                        self.main_window.video_loader_worker.thumbnail_ready.connect(partial(list_view_actions.add_media_thumbnail_to_target_videos_list, self.main_window))
                        self.main_window.video_loader_worker.start()
                    event.acceptProposedAction()
                    return True


        elif list_widget == self.main_window.inputFacesList or list_widget == self.main_window.inputFacesList.viewport():

            if event.type() == QtCore.QEvent.Type.MouseButtonPress:
                if event.button() == QtCore.Qt.MouseButton.LeftButton and not self.main_window.input_faces:
                    list_view_actions.select_input_face_images(self.main_window, 'folder')

            elif event.type() == QtCore.QEvent.Type.DragEnter:
                # Accept drag events with URLs
                if event.mimeData().hasUrls():

                    urls = event.mimeData().urls()
                    print("Drag: URLS", [url.toLocalFile() for url in urls])
                    event.acceptProposedAction()
                    return True
            # Handle the drop event
            elif event.type() == QtCore.QEvent.Type.Drop:

                if event.mimeData().hasUrls():
                    # Extract file paths
                    file_paths = []
                    for url in event.mimeData().urls():
                        url = url.toLocalFile()
                        if misc_helpers.is_image_file(url):
                            file_paths.append(url)
                        else:
                            print(f'{url} is not an Image file')
                    # print("Drop: URLS", [url.toLocalFile() for url in urls])
                    if file_paths:
                        self.main_window.input_faces_loader_worker = ui_workers.InputFacesLoaderWorker(main_window=self.main_window, folder_name=False, files_list=file_paths)
                        self.main_window.input_faces_loader_worker.thumbnail_ready.connect(partial(list_view_actions.add_media_thumbnail_to_source_faces_list, self.main_window))
                        self.main_window.input_faces_loader_worker.start()
                    event.acceptProposedAction()
                    return True
        return super().eventFilter(list_widget, event)