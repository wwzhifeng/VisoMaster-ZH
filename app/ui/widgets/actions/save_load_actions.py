import json
from pathlib import Path
import uuid
import copy
from functools import partial
from typing import TYPE_CHECKING, Dict

from PySide6 import QtWidgets
import numpy as np

from app.ui.widgets.actions import common_actions as common_widget_actions
from app.ui.widgets.actions import card_actions
from app.ui.widgets.actions import list_view_actions
from app.ui.widgets.actions import video_control_actions
from app.ui.widgets.actions import layout_actions
from app.ui.widgets import ui_workers
from app.helpers.typing_helper import ParametersTypes, MarkerTypes
import app.helpers.miscellaneous as misc_helpers

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow

def open_embeddings_from_file(main_window: 'MainWindow'):
    
    embedding_filename, _ = QtWidgets.QFileDialog.getOpenFileName(main_window, filter='JSON (*.json)', dir=misc_helpers.get_dir_of_file(main_window.loaded_embedding_filename))
    if embedding_filename:
        with open(embedding_filename, 'r') as embed_file: #pylint: disable=unspecified-encoding
            embeddings_list = json.load(embed_file)
            card_actions.clear_merged_embeddings(main_window)

            # Reset per ogni target face
            for _, target_face in main_window.target_faces.items():
                target_face.assigned_merged_embeddings = {}
                target_face.assigned_input_embedding = {}

            # Carica gli embedding dal file e crea il dizionario embedding_store
            for embed_data in embeddings_list:
                embedding_store = embed_data.get('embedding_store', {})
                # Converte ogni embedding in numpy array
                for recogn_model, embed in embedding_store.items():
                    embedding_store[recogn_model] = np.array(embed)

                # Passa l'intero embedding_store alla funzione
                list_view_actions.create_and_add_embed_button_to_list(
                    main_window, 
                    embed_data['name'], 
                    embedding_store,  # Passa l'intero embedding_store
                    embedding_id=str(uuid.uuid1().int)
                )

    main_window.loaded_embedding_filename = embedding_filename or main_window.loaded_embedding_filename

def save_embeddings_to_file(main_window: 'MainWindow', save_as=False):
    if not main_window.merged_embeddings:
        common_widget_actions.create_and_show_messagebox(main_window, 'Embeddings List Empty!', 'No Embeddings available to save', parent_widget=main_window)
        return

    # Definisce il nome del file di salvataggio
    embedding_filename = main_window.loaded_embedding_filename
    if not embedding_filename or not misc_helpers.is_file_exists(embedding_filename) or save_as:
        embedding_filename, _ = QtWidgets.QFileDialog.getSaveFileName(main_window, filter='JSON (*.json)')

    # Crea una lista di dizionari, ciascuno con il nome dell'embedding e il relativo embedding_store
    embeddings_list = [
        {
            'name': embed_button.embedding_name,
            'embedding_store': {k: v.tolist() for k, v in embed_button.embedding_store.items()}  # Converti gli embedding in liste
        }
        for embedding_id, embed_button in main_window.merged_embeddings.items()
    ]

    # Salva su file
    if embedding_filename:
        with open(embedding_filename, 'w') as embed_file: #pylint: disable=unspecified-encoding
            embeddings_as_json = json.dumps(embeddings_list, indent=4)  # Salva con indentazione per leggibilità
            embed_file.write(embeddings_as_json)

            # Mostra un messaggio di conferma
            common_widget_actions.create_and_show_toast_message(main_window, 'Embeddings Saved', f'Saved Embeddings to file: {embedding_filename}')

        main_window.loaded_embedding_filename = embedding_filename

# This method is used to convert the data type of Parameters Dict
# Parameters are converted to dict when serializing to JSON
# Parameters are converted to ParametersDict when reading from JSON 
def convert_parameters_to_supported_type(main_window: 'MainWindow', parameters: dict|ParametersTypes, convert_type: dict|misc_helpers.ParametersDict):
    if convert_type==dict:
        parameters = parameters.data
    elif convert_type==misc_helpers.ParametersDict:
        parameters = misc_helpers.ParametersDict(parameters, main_window.default_parameters)
    return parameters

def convert_markers_to_supported_type(main_window: 'MainWindow', markers: MarkerTypes, convert_type: dict|misc_helpers.ParametersDict):
    # Convert Parameters inside the markers from ParametersDict to dict
    for _,marker_data in markers.items():
        for target_face_id, target_parameters in marker_data['parameters'].items():
            marker_data['parameters'][target_face_id] = convert_parameters_to_supported_type(main_window, target_parameters, convert_type)
    return markers

def save_current_parameters_and_control(main_window: 'MainWindow', face_id):
    data_filename, _ = QtWidgets.QFileDialog.getSaveFileName(main_window, filter='JSON (*.json)')
    data = {
        'parameters': convert_parameters_to_supported_type(main_window, main_window.parameters[face_id], dict),
        'control': main_window.control.copy(),
    }

    if data_filename:
        with open(data_filename, 'w') as data_file: #pylint: disable=unspecified-encoding
            data_as_json = json.dumps(data, indent=4)  # Salva con indentazione per leggibilità
            data_file.write(data_as_json)

def load_parameters_and_settings(main_window: 'MainWindow', face_id, load_settings=False):
    data_filename, _ = QtWidgets.QFileDialog.getOpenFileName(main_window, filter='JSON (*.json)')
    if data_filename:
        with open(data_filename, 'r') as data_file: #pylint: disable=unspecified-encoding
            data = json.load(data_file)
            main_window.parameters[face_id] = convert_parameters_to_supported_type(main_window, data['parameters'].copy(), misc_helpers.ParametersDict)
            if main_window.selected_target_face_id == face_id:
                common_widget_actions.set_widgets_values_using_face_id_parameters(main_window, face_id)
            if load_settings:
                main_window.control.update(data['control'])
                common_widget_actions.set_control_widgets_values(main_window)
            common_widget_actions.refresh_frame(main_window)

def load_saved_workspace(main_window: 'MainWindow', data_filename: str|bool = False):
    if not data_filename:
        data_filename, _ = QtWidgets.QFileDialog.getOpenFileName(main_window, filter='JSON (*.json)')
    # Check if File exists (In cases when filename is passed as function argument instead of from the file picker)
    if not Path(data_filename).is_file():
        data_filename = False
    if data_filename:
        with open(data_filename, 'r') as data_file: #pylint: disable=unspecified-encoding
            data = json.load(data_file)
            list_view_actions.clear_stop_loading_input_media(main_window)
            list_view_actions.clear_stop_loading_target_media(main_window)
            main_window.target_videos = {}
            card_actions.clear_input_faces(main_window)
            card_actions.clear_target_faces(main_window)
            card_actions.clear_merged_embeddings(main_window)

            # Add target medias
            target_medias_data = data['target_medias_data']
            target_medias_files_list = []
            target_media_ids = []
            for media_data in target_medias_data:
                target_medias_files_list.append(media_data['media_path'])
                target_media_ids.append(media_data['media_id'])

            main_window.video_loader_worker = ui_workers.TargetMediaLoaderWorker(main_window=main_window, folder_name=False, files_list=target_medias_files_list, media_ids=target_media_ids)
            main_window.video_loader_worker.thumbnail_ready.connect(partial(list_view_actions.add_media_thumbnail_to_target_videos_list, main_window))
            main_window.video_loader_worker.run()

            # Select target media
            selected_media_id = data['selected_media_id']
            if selected_media_id is not False and main_window.target_videos.get(selected_media_id):
                main_window.target_videos[selected_media_id].click()

            # Add input faces (imgs)
            input_media_paths, input_face_ids = [], []
            for face_id, input_face_data in data['input_faces_data'].items():
                input_media_paths.append(input_face_data['media_path'])
                input_face_ids.append(face_id)
            main_window.input_faces_loader_worker = ui_workers.InputFacesLoaderWorker(main_window=main_window, folder_name=False, files_list=input_media_paths, face_ids=input_face_ids)
            main_window.input_faces_loader_worker.thumbnail_ready.connect(partial(list_view_actions.add_media_thumbnail_to_source_faces_list, main_window))
            main_window.input_faces_loader_worker.finished.connect(partial(common_widget_actions.refresh_frame, main_window))
            #Use run() instead of start(), as we dont want it running in a different thread as it could create synchronisation issues in the steps below
            main_window.input_faces_loader_worker.run() 

            # Add embeddings
            embeddings_data = data['embeddings_data']
            for embedding_id, embedding_data in embeddings_data.items():
                embedding_store = {embed_model: np.array(embedding) for embed_model, embedding in embedding_data['embedding_store'].items()}
                embedding_name = embedding_data['embedding_name']
                list_view_actions.create_and_add_embed_button_to_list(main_window, embedding_name, embedding_store, embedding_id=embedding_id)

            # Add target_faces
            for face_id, target_face_data in data['target_faces_data'].items():
                cropped_face = np.array(target_face_data['cropped_face']).astype('uint8')
                pixmap = common_widget_actions.get_pixmap_from_frame(main_window, cropped_face)
                embedding_store: Dict[str, np.ndarray] = {embed_model: np.array(embedding) for embed_model, embedding in target_face_data['embedding_store'].items()}
                list_view_actions.add_media_thumbnail_to_target_faces_list(main_window, cropped_face, embedding_store, pixmap, face_id)
                main_window.parameters[face_id] = convert_parameters_to_supported_type(main_window, main_window.parameters[face_id], misc_helpers.ParametersDict)

                # Set assigned embeddinng buttons
                embed_buttons = main_window.merged_embeddings
                assigned_merged_embeddings: list = target_face_data['assigned_merged_embeddings']
                for assigned_merged_embedding_id in assigned_merged_embeddings:
                    main_window.target_faces[face_id].assigned_merged_embeddings[assigned_merged_embedding_id] = embed_buttons[assigned_merged_embedding_id].embedding_store

                # Set assigned input face buttons
                assigned_input_faces: list = target_face_data['assigned_input_faces']
                for assigned_input_face_id in assigned_input_faces:
                    main_window.target_faces[face_id].assigned_input_faces[assigned_input_face_id] = main_window.input_faces[assigned_input_face_id].embedding_store
                
                # Set assigned input embedding (Input face + merged embeddings)
                assigned_input_embedding = {embed_model: np.array(embedding) for embed_model, embedding in target_face_data['assigned_input_embedding'].items()}
                main_window.target_faces[face_id].assigned_input_embedding = assigned_input_embedding
                # main_window.control = target_face_data['control']

            # Load control (settings)
            control = data['control']
            for control_name, control_value in control.items():
                main_window.control[control_name] = control_value

            # Add markers
            video_control_actions.remove_all_markers(main_window)

            # Convert params to ParametersDict
            data['markers'] = convert_markers_to_supported_type(main_window, data['markers'], misc_helpers.ParametersDict)
        
            for marker_position, marker_data in data['markers'].items():
                video_control_actions.add_marker(main_window, marker_data['parameters'], marker_data['control'], int(marker_position))
            # main_window.videoSeekSlider.setValue(0)
            # video_control_actions.update_widget_values_from_markers(main_window, 0)

            # Set target media and input faces folder names
            main_window.last_target_media_folder_path = data.get('last_target_media_folder_path','')
            main_window.last_input_media_folder_path = data.get('last_input_media_folder_path','')
            main_window.loaded_embedding_filename = data.get('loaded_embedding_filename', '')
            common_widget_actions.set_control_widgets_values(main_window)
            # Set output folder
            common_widget_actions.create_control(main_window, 'OutputMediaFolder', control['OutputMediaFolder'])
            main_window.outputFolderLineEdit.setText(control['OutputMediaFolder'])

            layout_actions.fit_image_to_view_onchange(main_window)

            if main_window.target_faces:
                list(main_window.target_faces.values())[0].click()
            else:
                main_window.current_widget_parameters = data.get('current_widget_parameters', main_window.default_parameters.copy())
                main_window.current_widget_parameters = misc_helpers.ParametersDict(main_window.current_widget_parameters, main_window.default_parameters)
                common_widget_actions.set_widgets_values_using_face_id_parameters(main_window, face_id=False) 
        
def save_current_workspace(main_window: 'MainWindow', data_filename:str|bool = False):
    target_faces_data = {}
    embeddings_data = {}
    input_faces_data = {}
    for face_id, input_face in main_window.input_faces.items():
        input_faces_data[face_id] = {'media_path': input_face.media_path}
    for face_id, target_face in main_window.target_faces.items():
        target_faces_data[face_id] = {
            'cropped_face': target_face.cropped_face.tolist(), 
            'embedding_store': {embed_model: embedding.tolist() for embed_model, embedding in target_face.embedding_store.items()},
            'parameters': main_window.parameters[face_id].data.copy(), #Store the current parameters. This will be overriden when loading the workspace, if there are markers for the video.
            'control': main_window.control.copy(), #Store the current control settings. This will be overriden when loading the workspace, if there are markers for the video.
            'assigned_input_faces': [input_face_id for input_face_id in target_face.assigned_input_faces.keys()],
            'assigned_merged_embeddings': [embedding_id for embedding_id in target_face.assigned_merged_embeddings.keys()],
            'assigned_input_embedding': {embed_model: embedding.tolist() for embed_model, embedding in target_face.assigned_input_embedding.items()}
            }
    for embedding_id, embed_button in main_window.merged_embeddings.items():
        embeddings_data[embedding_id] = {
            'embedding_store': {embed_model: embedding.tolist() for embed_model,embedding in embed_button.embedding_store.items()}, 
            'embedding_name': embed_button.embedding_name}
    
    target_medias_data = [{'media_id': media_id, 'media_path': target_media.media_path}  for media_id,target_media in main_window.target_videos.items() if not target_media.is_webcam]
    selected_media_id = main_window.selected_video_button.media_id if main_window.selected_video_button else False
    markers = copy.deepcopy(main_window.markers)
    # Convert params to dict
    markers = convert_markers_to_supported_type(main_window, markers, dict)

    save_data = {
        'selected_media_id': selected_media_id,
        'target_medias_data': target_medias_data,
        'target_faces_data': target_faces_data,
        'embeddings_data': embeddings_data,
        'input_faces_data': input_faces_data,
        'markers': markers,
        'control': main_window.control,
        'last_target_media_folder_path': main_window.last_target_media_folder_path,
        'last_input_media_folder_path': main_window.last_input_media_folder_path,
        'loaded_embedding_filename': main_window.loaded_embedding_filename,
        'current_widget_parameters': convert_parameters_to_supported_type(main_window, main_window.current_widget_parameters, dict)
    }
    if not data_filename:
        data_filename, _ = QtWidgets.QFileDialog.getSaveFileName(main_window, filter='JSON (*.json)')

    if data_filename:
        with open(data_filename, 'w') as data_file: #pylint: disable=unspecified-encoding
            data_as_json = json.dumps(save_data, indent=4)  # Salva con indentazione per leggibilità
            data_file.write(data_as_json)