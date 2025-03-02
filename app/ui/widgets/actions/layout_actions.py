from typing import TYPE_CHECKING
from functools import partial

from PySide6 import QtWidgets, QtCore
if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow
from app.ui.widgets.actions import common_actions as common_widget_actions
from app.ui.widgets.actions import graphics_view_actions
from app.ui.widgets.actions import list_view_actions
from app.ui.widgets.actions import save_load_actions
from app.ui.widgets.actions import video_control_actions
from app.ui.widgets import widget_components
# from app.UI.Widgets.WidgetComponents import *
from app.helpers.typing_helper import LayoutDictTypes

def add_widgets_to_tab_layout(main_window: 'MainWindow', LAYOUT_DATA: LayoutDictTypes, layoutWidget: QtWidgets.QVBoxLayout, data_type='parameter'):
    layout = QtWidgets.QVBoxLayout()
    layout.setContentsMargins(0, 0, 10, 0)  # Adjust left margin (20px in this example)
    scroll_area = QtWidgets.QScrollArea()
    scroll_area.setWidgetResizable(True)
    scroll_content = QtWidgets.QWidget()
    scroll_content.setLayout(layout)
    scroll_area.setWidget(scroll_content)
    scroll_area.setFrameShape(QtWidgets.QFrame.NoFrame)

    def add_horizontal_layout_to_category(category_layout: QtWidgets.QFormLayout, *widgets):
        # Create a horizontal layout
        horizontal_layout = QtWidgets.QHBoxLayout()
        
        for widget in widgets:
            horizontal_layout.addWidget(widget)  # Add the toggle button
        category_layout.addRow(horizontal_layout)  # Add the horizontal layout to the form layout
        return horizontal_layout

    for category, widgets in LAYOUT_DATA.items():
        group_box = widget_components.FormGroupBox(main_window, title=category)
        category_layout = QtWidgets.QFormLayout()
        group_box.setLayout(category_layout)

        for widget_name, widget_data in widgets.items():
            spacing_level = widget_data['level']
            label = QtWidgets.QLabel(widget_data['label'])
            label.setToolTip(widget_data['help'])

            # Create a horizontal layout for the toggle button and its label
            if 'Toggle' in widget_name:
                widget = widget_components.ToggleButton(label=widget_data['label'], widget_name=widget_name, group_layout_data=widgets, label_widget=label, main_window=main_window)
                widget.setChecked(widget_data['default'])
                widget.reset_default_button = widget_components.ParameterResetDefaultButton(related_widget=widget)

                # Create a horizontal layout
                horizontal_layout = add_horizontal_layout_to_category(category_layout, widget, label, widget.reset_default_button)

                if data_type=='parameter':
                    # Initialize parameter value
                    common_widget_actions.create_default_parameter(main_window, widget_name, widget_data['default'])
                else:
                    common_widget_actions.create_control(main_window, widget_name, widget_data['default'])
                # Set onclick function for toggle button
                def onchange(toggle_widget: widget_components.ToggleButton, toggle_widget_name, widget_data: dict, *args):
                    toggle_state = toggle_widget.isChecked()
                    if data_type=='parameter':
                        common_widget_actions.update_parameter(main_window, toggle_widget_name, toggle_state, enable_refresh_frame=toggle_widget.enable_refresh_frame, exec_function=widget_data.get('exec_function'), exec_function_args=widget_data.get('exec_function_args', []))    
                    elif data_type=='control':
                        common_widget_actions.update_control(main_window, toggle_widget_name, toggle_state, exec_function=widget_data.get('exec_function'), exec_function_args=widget_data.get('exec_function_args', []))
                widget.toggled.connect(partial(onchange, widget, widget_name, widget_data))

            elif 'Selection' in widget_name:
                widget = widget_components.SelectionBox(label=widget_data['label'], widget_name=widget_name, group_layout_data=widgets, label_widget=label, main_window=main_window, default_value=widget_data['default'], selection_values=widget_data['options'])
                if callable(widget_data['options']):
                    widget.addItems(widget_data['options']())
                    widget.setCurrentText(widget_data['default']())
                else:
                    widget.addItems(widget_data['options'])
                    widget.setCurrentText(widget_data['default'])

                widget.reset_default_button = widget_components.ParameterResetDefaultButton(related_widget=widget)

                horizontal_layout = add_horizontal_layout_to_category(category_layout, label, widget, widget.reset_default_button)


                if data_type=='parameter':
                    # Initialize parameter value
                    common_widget_actions.create_default_parameter(main_window, widget_name, widget_data['default'] if not callable(widget_data['default']) else widget_data['default']())
                else:
                    common_widget_actions.create_control(main_window, widget_name, widget_data['default'] if not callable(widget_data['default']) else widget_data['default']())
                # Set onchange function for select box (Selected value is passed by the signal)
                def onchange(selection_widget: widget_components.SelectionBox, selection_widget_name, widget_data: dict, selected_value=False):
                    # selected_value = selection_widget.currentText()
                    if data_type=='parameter':
                        common_widget_actions.update_parameter(main_window, selection_widget_name, selected_value, enable_refresh_frame=selection_widget.enable_refresh_frame)
                    elif data_type=='control':
                        common_widget_actions.update_control(main_window, selection_widget_name, selected_value, exec_function=widget_data.get('exec_function'), exec_function_args=widget_data.get('exec_function_args', []))
                widget.currentTextChanged.connect(partial(onchange, widget, widget_name, widget_data))

            elif 'DecimalSlider' in widget_name:
                widget = widget_components.ParameterDecimalSlider(
                    label=widget_data['label'], 
                    widget_name=widget_name, 
                    group_layout_data=widgets, 
                    label_widget=label, 
                    min_value=float(widget_data['min_value']),  # Ensure min_value is float
                    max_value=float(widget_data['max_value']),  # Ensure max_value is float
                    default_value=float(widget_data['default']),  # Ensure default_value is float
                    decimals=int(widget_data['decimals']),
                    step_size=float(widget_data['step']),
                    main_window=main_window
                )
                # Use the new ParameterLineDecimalEdit class
                widget.line_edit = widget_components.ParameterLineDecimalEdit(
                    min_value=float(widget_data['min_value']), 
                    max_value=float(widget_data['max_value']), 
                    default_value=str(widget_data['default']),
                    decimals=int(widget_data['decimals']),  # Ensure it uses decimals place for consistency
                    step_size=float(widget_data['step']),
                    fixed_width=48,
                    max_length=7 if int(widget_data['decimals']) > 1 else 5
                )
                widget.reset_default_button = widget_components.ParameterResetDefaultButton(related_widget=widget)

                # Layout for widgets
                horizontal_layout = add_horizontal_layout_to_category(category_layout, label, widget, widget.line_edit, widget.reset_default_button)


                if data_type=='parameter':
                    # Initialize parameter value
                    common_widget_actions.create_default_parameter(main_window, widget_name, float(widget_data['default']))

                else:
                    common_widget_actions.create_control(main_window, widget_name, float(widget_data['default']))

                # When slider value changes
                def onchange_slider(slider_widget: widget_components.ParameterDecimalSlider, slider_widget_name, widget_data: dict, new_value=False):
                    # Update the slider text box value too
                    actual_value = slider_widget.value()  # Get float value from the slider
                    if data_type=='parameter':
                        common_widget_actions.update_parameter(main_window, slider_widget_name, actual_value, enable_refresh_frame=slider_widget.enable_refresh_frame)
                    elif data_type=='control':
                        common_widget_actions.update_control(main_window, slider_widget_name, actual_value, exec_function=widget_data.get('exec_function'), exec_function_args=widget_data.get('exec_function_args', []))
                    slider_widget.line_edit.set_value(actual_value)  # Update the line edit with the actual value

                # Invece di collegare direttamente onchange_slider, fallo dopo il debounce
                widget.debounce_timer.timeout.connect(partial(onchange_slider, widget, widget_name, widget_data))

                # When line edit value changes
                def onchange_line_edit(slider_widget: widget_components.ParameterDecimalSlider, slider_widget_name: str, widget_data: dict, new_value=False):
                    """Handle changes in the line edit and update the slider accordingly."""
                    if not new_value:
                        new_value = 0.0

                    try:
                        # Convert the text box input to float
                        new_value = float(new_value)
                    except ValueError:
                        # If the conversion fails, reset the line edit to the current slider value
                        new_value = slider_widget.value()

                    # Prevent text box value from exceeding the slider range or going below minimum
                    if new_value > (slider_widget.max_value / slider_widget.scale_factor):
                        new_value = slider_widget.max_value / slider_widget.scale_factor
                    elif new_value < (slider_widget.min_value / slider_widget.scale_factor):
                        new_value = slider_widget.min_value / slider_widget.scale_factor

                    # Update the slider's internal value
                    slider_widget.setValue(new_value)
                    # Update the line_edit text to reflect the current value
                    slider_widget.line_edit.set_value(new_value)

                    if data_type=='parameter':
                        common_widget_actions.update_parameter(main_window, slider_widget_name, new_value, enable_refresh_frame=slider_widget.enable_refresh_frame)
                    elif data_type=='control':
                        common_widget_actions.update_control(main_window, slider_widget_name, new_value, exec_function=widget_data.get('exec_function'), exec_function_args=widget_data.get('exec_function_args', []))

                widget.line_edit.textChanged.connect(partial(onchange_line_edit, widget, widget_name, widget_data))
 
            elif 'Slider' in widget_name:
                widget = widget_components.ParameterSlider(label=widget_data['label'], widget_name=widget_name, group_layout_data=widgets, label_widget=label, min_value=widget_data['min_value'], max_value=widget_data['max_value'], default_value=widget_data['default'], step_size=widget_data['step'], main_window=main_window)
                widget.line_edit = widget_components.ParameterLineEdit(min_value=int(widget_data['min_value']), max_value=int(widget_data['max_value']), default_value=widget_data['default'])
                widget.reset_default_button = widget_components.ParameterResetDefaultButton(related_widget=widget)
                
                horizontal_layout = add_horizontal_layout_to_category(category_layout, label, widget, widget.line_edit, widget.reset_default_button)


                if data_type=='parameter':
                    # Initialize parameter value
                    common_widget_actions.create_default_parameter(main_window, widget_name, int(widget_data['default']))
                else:
                    common_widget_actions.create_control(main_window, widget_name, int(widget_data['default']))

                # When slider value is change
                def onchange_slider(slider_widget: widget_components.ParameterSlider, slider_widget_name, widget_data: dict, new_value=False):
                    if data_type=='parameter':
                        common_widget_actions.update_parameter(main_window, slider_widget_name, new_value, enable_refresh_frame=slider_widget.enable_refresh_frame)
                    elif data_type=='control':
                        common_widget_actions.update_control(main_window, slider_widget_name, new_value, exec_function=widget_data.get('exec_function'), exec_function_args=widget_data.get('exec_function_args', []))
                    # Update the slider text box value too
                    slider_widget.line_edit.setText(str(new_value))

                # Invece di collegare direttamente onchange_slider, fallo dopo il debounce
                widget.debounce_timer.timeout.connect(partial(onchange_slider, widget, widget_name, widget_data))

                # When slider textbox value is changed
                def onchange_line_edit(slider_widget: widget_components.ParameterSlider, slider_widget_name, widget_data, new_value=False):
                    """Handle changes in the line edit and update the slider accordingly."""
                    if not new_value:
                        new_value = 0
                    try:
                        # Prova a convertire il valore in intero, se fallisce usa il valore corrente dello slider
                        new_value = int(new_value)
                    except ValueError:
                        # Se non è possibile convertire, mantieni il valore corrente dello slider
                        new_value = slider_widget.value()
                    # Prevent the text box value from going above the maximum value of the slider
                    if new_value > slider_widget.max_value:
                        new_value = slider_widget.max_value
                    elif new_value < slider_widget.min_value:
                        new_value = slider_widget.min_value

                    slider_widget.line_edit.set_value(new_value)
                    slider_widget.setValue(int(new_value)) #Update the value of slider too
                    if data_type=='parameter':
                        common_widget_actions.update_parameter(main_window, slider_widget_name, new_value, enable_refresh_frame=slider_widget.enable_refresh_frame)
                    elif data_type=='control':
                        common_widget_actions.update_control(main_window, slider_widget_name, new_value, exec_function=widget_data.get('exec_function'), exec_function_args=widget_data.get('exec_function_args', []))

                widget.line_edit.textChanged.connect(partial(onchange_line_edit, widget, widget_name, widget_data))

            elif 'Text' in widget_name:
                widget = widget_components.ParameterText(label=widget_data['label'], widget_name=widget_name, group_layout_data=widgets, label_widget=label, default_value=widget_data['default'], fixed_width=widget_data['width'], main_window=main_window, data_type=data_type, exec_function=widget_data.get('exec_function'), exec_function_args=widget_data.get('exec_function_args', []))
                widget.reset_default_button = widget_components.ParameterResetDefaultButton(related_widget=widget)

                horizontal_layout = add_horizontal_layout_to_category(category_layout, label, widget, widget.reset_default_button)


                if data_type=='parameter':
                    # Initialize parameter value
                    common_widget_actions.create_default_parameter(main_window, widget_name, widget_data['default'])

                else:
                    common_widget_actions.create_control(main_window, widget_name, widget_data['default'])

                 # Handle 'Enter' key press to confirm input
                def on_enter_pressed(text_widget: widget_components.ParameterText, text_widget_name):
                    # Logic to confirm input or trigger any action when Enter is pressed
                    new_value = text_widget.text()  # Get the current text value
                    if data_type == 'parameter':
                        common_widget_actions.update_parameter(main_window, text_widget_name, new_value, enable_refresh_frame=text_widget.enable_refresh_frame)
                    else:
                        common_widget_actions.update_control(main_window, text_widget_name, new_value, exec_function=widget_data.get('exec_function'), exec_function_args=widget_data.get('exec_function_args', [])) # pylint: disable=cell-var-from-loop
                
                # Connect the 'returnPressed' signal to the on_enter_pressed function
                widget.returnPressed.connect(partial(on_enter_pressed, widget, widget_name))

            horizontal_layout.setContentsMargins(spacing_level * 10, 0, 0, 0)

            main_window.parameter_widgets[widget_name] = widget
        # category_layout.setContentsMargins(20,20,20,20)
        category_layout.setVerticalSpacing(2)
        category_layout.setHorizontalSpacing(2)

        layout.addWidget(group_box)

    layoutWidget.addWidget(scroll_area)

    # Default show/hide widgets
    for category, widgets in LAYOUT_DATA.items():
        for widget_name, widget_data in widgets.items():
            widget = main_window.parameter_widgets[widget_name]
            common_widget_actions.show_hide_related_widgets(main_window, widget, widget_name)

def show_hide_faces_panel(main_window: 'MainWindow', checked):
    if checked:
        main_window.facesPanelGroupBox.show()
    else:
        main_window.facesPanelGroupBox.hide()
    fit_image_to_view_onchange(main_window)

def show_hide_input_target_media_panel(main_window: 'MainWindow', checked):
    if checked:
        main_window.input_Target_DockWidget.show()
    else:
        main_window.input_Target_DockWidget.hide()
    fit_image_to_view_onchange(main_window)

def show_hide_parameters_panel(main_window: 'MainWindow', checked):
    if checked:
        main_window.controlOptionsDockWidget.show()
    else:
        main_window.controlOptionsDockWidget.hide()
    fit_image_to_view_onchange(main_window)

def fit_image_to_view_onchange(main_window: 'MainWindow', *args):
    pixmap_items = main_window.scene.items()
    if pixmap_items:
        pixmap_item = pixmap_items[0]
        scene_rect = pixmap_item.boundingRect()
        QtCore.QTimer.singleShot(0, partial(graphics_view_actions.fit_image_to_view, main_window, pixmap_item, scene_rect))

def set_up_menu_actions(main_window: 'MainWindow'):
    main_window.actionLoad_SavedWorkspace.triggered.connect(partial(save_load_actions.load_saved_workspace, main_window,))
    main_window.actionSave_CurrentWorkspace.triggered.connect(partial(save_load_actions.save_current_workspace, main_window,))

    main_window.actionOpen_Videos_Folder.triggered.connect(partial(list_view_actions.select_target_medias, main_window, 'folder'))
    main_window.actionOpen_Video_Files.triggered.connect(partial(list_view_actions.select_target_medias, main_window, 'files'))
    main_window.actionLoad_Source_Image_Files.triggered.connect(partial(list_view_actions.select_input_face_images, main_window, 'files'))
    main_window.actionLoad_Source_Images_Folder.triggered.connect(partial(list_view_actions.select_input_face_images, main_window, 'folder'))
    main_window.actionLoad_Embeddings.triggered.connect(partial(save_load_actions.open_embeddings_from_file, main_window))
    main_window.actionSave_Embeddings.triggered.connect(partial(save_load_actions.save_embeddings_to_file, main_window))
    main_window.actionSave_Embeddings_As.triggered.connect(partial(save_load_actions.save_embeddings_to_file, main_window))
    main_window.actionView_Fullscreen_F11.triggered.connect(partial(video_control_actions.view_fullscreen, main_window))

def disable_all_parameters_and_control_widget(main_window: 'MainWindow'):
    # Disable all bottom buttons
    main_window.saveImageButton.setDisabled(True)
    main_window.findTargetFacesButton.setDisabled(True)
    main_window.clearTargetFacesButton.setDisabled(True)
    main_window.swapfacesButton.setDisabled(True)
    main_window.editFacesButton.setDisabled(True)
    main_window.openEmbeddingButton.setDisabled(True)
    main_window.saveEmbeddingButton.setDisabled(True)
    main_window.saveEmbeddingAsButton.setDisabled(True)

    # Disable all video control buttons
    main_window.videoSeekSlider.setDisabled(True)
    main_window.addMarkerButton.setDisabled(True)
    main_window.removeMarkerButton.setDisabled(True)
    main_window.nextMarkerButton.setDisabled(True)
    main_window.previousMarkerButton.setDisabled(True)
    main_window.frameAdvanceButton.setDisabled(True)
    main_window.frameRewindButton.setDisabled(True)

    # Enable compare checkboxes
    main_window.faceCompareCheckBox.setDisabled(True)
    main_window.faceMaskCheckBox.setDisabled(True)


    # Disable list items
    for _, embed_button in main_window.merged_embeddings.items():
        embed_button.setDisabled(True)
    for _, target_media_button in main_window.target_videos.items():
        target_media_button.setDisabled(True)
    for _, input_face_button in main_window.input_faces.items():
        input_face_button.setDisabled(True)
    for _, target_face_button in main_window.target_faces.items():
        target_face_button.setDisabled(True)


    # Disable parameters and controls dict widgets
    for _, widget in main_window.parameter_widgets.items():
        widget.setDisabled(True)
        widget.reset_default_button.setDisabled(True)
        widget.label_widget.setDisabled(True)
        if widget.line_edit:
            widget.line_edit.setDisabled(True)

def enable_all_parameters_and_control_widget(main_window: 'MainWindow'):
    # Enable all bottom buttons
    main_window.saveImageButton.setDisabled(False)
    main_window.findTargetFacesButton.setDisabled(False)
    main_window.clearTargetFacesButton.setDisabled(False)
    main_window.swapfacesButton.setDisabled(False)
    main_window.editFacesButton.setDisabled(False)
    main_window.openEmbeddingButton.setDisabled(False)
    main_window.saveEmbeddingButton.setDisabled(False)
    main_window.saveEmbeddingAsButton.setDisabled(False)

    # Enable all video control buttons
    main_window.videoSeekSlider.setDisabled(False)
    main_window.addMarkerButton.setDisabled(False)
    main_window.removeMarkerButton.setDisabled(False)
    main_window.nextMarkerButton.setDisabled(False)
    main_window.previousMarkerButton.setDisabled(False)
    main_window.frameAdvanceButton.setDisabled(False)
    main_window.frameRewindButton.setDisabled(False)

    # Enable compare checkboxes
    main_window.faceCompareCheckBox.setDisabled(False)
    main_window.faceMaskCheckBox.setDisabled(False)

    # Enable list items
    for _, embed_button in main_window.merged_embeddings.items():
        embed_button.setDisabled(False)
    for _, target_media_button in main_window.target_videos.items():
        target_media_button.setDisabled(False)
    for _, input_face_button in main_window.input_faces.items():
        input_face_button.setDisabled(False)
    for _, target_face_button in main_window.target_faces.items():
        target_face_button.setDisabled(False)

    # Enable parameters and controls dict widgets
    for _, widget in main_window.parameter_widgets.items():
        widget.setDisabled(False)
        widget.reset_default_button.setDisabled(False)
        widget.label_widget.setDisabled(False)
        if widget.line_edit:
            widget.line_edit.setDisabled(False)