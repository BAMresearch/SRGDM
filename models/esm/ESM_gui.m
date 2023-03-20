function varargout = ESM_gui(varargin)
% ESM_GUI MATLAB code for ESM_gui.fig
%      ESM_GUI, by itself, creates a new ESM_GUI or raises the existing
%      singleton*.
%
%      H = ESM_GUI returns the handle to a new ESM_GUI or the handle to
%      the existing singleton*.
%
%      ESM_GUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in ESM_GUI.M with the given input arguments.
%
%      ESM_GUI('Property','Value',...) creates a new ESM_GUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before ESM_gui_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to ESM_gui_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help ESM_gui

% Last Modified by GUIDE v2.5 11-Aug-2016 13:59:25

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @ESM_gui_OpeningFcn, ...
                   'gui_OutputFcn',  @ESM_gui_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before ESM_gui is made visible.
function ESM_gui_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to ESM_gui (see VARARGIN)

% Choose default command line output for ESM_gui
handles.output = hObject;

% setting up simulator
simC = ESM_initializeSim('SimpleScenario');

% Original config
simC = ESM_addSensor(simC, [8, 2]);
simC = ESM_addSensor(simC, [14, 8]);
simC = ESM_addSensor(simC, [8, 14]);
simC = ESM_addSensor(simC, [2, 8]);

%inner corner config
% simC = ESM_addSensor(simC, [8, 6]);
% simC = ESM_addSensor(simC, [10, 8]);
% simC = ESM_addSensor(simC, [8, 10]);
% simC = ESM_addSensor(simC, [6, 8]);

simC = ESM_simpleInterpolation(simC);
simC = ESM_eqsInterpolation(simC);
simC = ESM_esmInterpolation(simC);
handles.SimC = simC;

handles.PlayState = 0;

% setting up gui
axes(handles.axesTrueDist);
colormap(hot);
axis off;
axis equal;

axes(handles.axesEstimatedDist);
colormap(winter);
axis off;
axis equal;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes ESM_gui wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = ESM_gui_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on selection change in popEstimatorChoice.
function popEstimatorChoice_Callback(hObject, eventdata, handles)
% hObject    handle to popEstimatorChoice (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns popEstimatorChoice contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popEstimatorChoice


% --- Executes during object creation, after setting all properties.
function popEstimatorChoice_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popEstimatorChoice (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in popErrorSensor.
function popErrorSensor_Callback(hObject, eventdata, handles)
% hObject    handle to popErrorSensor (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns popErrorSensor contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popErrorSensor


% --- Executes during object creation, after setting all properties.
function popErrorSensor_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popErrorSensor (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushPlay.
function pushPlay_Callback(hObject, eventdata, handles)
% hObject    handle to pushPlay (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

simC = handles.SimC;

if handles.PlayState == 0
    set(handles.pushPlay, 'String', '||');
    handles.PlayState = 1;
    guidata(hObject, handles);
    
    while handles.PlayState == 1
        handles = guidata(hObject);
        
        % Check the source activity settings
        sourceActivity = [0 0 0];
        
        if get(handles.checkboxSource1, 'Value') > 0
            sourceActivity(1) = get(handles.sliderSource1, 'Value');
        else
            sourceActivity(1) = 0;
        end % if
        
        if get(handles.checkboxSource3, 'Value') > 0
            sourceActivity(3) = get(handles.sliderSource3, 'Value');
        else
            sourceActivity(3) = 0;
        end % if
        
        if get(handles.checkboxSource2, 'Value') > 0
            sourceActivity(2) = get(handles.sliderSource2, 'Value');
        else
            sourceActivity(2) = 0;
        end % if
                
        
        simC = ESM_simulateStep(simC, sourceActivity);
        simC = ESM_simulateSensor(simC);
        simC = ESM_simpleIntEstimate(simC);
        
        % Top left display
        axes(handles.axesTrueDist);
        image(simC.State./18);
        axis off;
        axis equal;
        
        % Bottom left display
        axes(handles.axesEstimatedDist);
        if get(handles.popEstimatorChoice, 'Value') == 1
            image(simC.StateSimpleEstimate./18);
        elseif get(handles.popEstimatorChoice, 'Value') == 2
            image(simC.StateEqsEstimate./18);
        elseif get(handles.popEstimatorChoice, 'Value') == 3
            image(simC.StateEsmEstimate./18);
        end % if
        axis off;
        axis equal;
        
        axes(handles.axesErrorSensors);
        if get(handles.popErrorSensor, 'Value') == 1
            plot(simC.ErrorSimpleEstimate); hold on;
            plot(simC.ErrorEqsEstimate, 'r');
            plot(simC.ErrorEsmEstimate, 'g');
            hold off;
        else
            plot(simC.Sensors(get(handles.popErrorSensor, 'Value')-1).SensorReadings(max(1, length(simC.Sensors(get(handles.popErrorSensor, 'Value')-1).SensorReadings)-199):end) ,'r'); 
            hold on;
            plot(simC.Sensors(get(handles.popErrorSensor, 'Value')-1).TrueReadings(max(1, length(simC.Sensors(get(handles.popErrorSensor, 'Value')-1).TrueReadings)-199):end) ,'k')
            hold off;
        end % if
        
      %  figure(h); subplot(1,2,1); image(simC.State./18); subplot(1,2,2); image(simC.StateSimpleEstimate./18); 
        
      varNorm = 1000000000;
      meanNorm = 100000;
      
        set(handles.editSimpleMSE, 'String', num2str(mean(simC.ErrorSimpleEstimate)/meanNorm));
        set(handles.editSimpleVAR, 'String', num2str(var(simC.ErrorSimpleEstimate)/varNorm));
        set(handles.editEqsMSE, 'String', num2str(mean(simC.ErrorEqsEstimate)/meanNorm));
        set(handles.editEqsVAR, 'String', num2str(var(simC.ErrorEqsEstimate)/varNorm));
        set(handles.editEsmMSE, 'String', num2str(mean(simC.ErrorEsmEstimate)/meanNorm));
        set(handles.editEsmVAR, 'String', num2str(var(simC.ErrorEsmEstimate)/varNorm));
        drawnow; 
        pause(str2double(get(handles.editSimDelay,'String')))
        %pause(0.05);
    end % while
else
    set(handles.pushPlay, 'String', '>');
    handles.PlayState = 0;
    guidata(hObject, handles);
end % if
handles.SimC = simC;
guidata(hObject, handles);


function editSimDelay_Callback(hObject, eventdata, handles)
% hObject    handle to editSimDelay (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of editSimDelay as text
%        str2double(get(hObject,'String')) returns contents of editSimDelay as a double


% --- Executes during object creation, after setting all properties.
function editSimDelay_CreateFcn(hObject, eventdata, handles)
% hObject    handle to editSimDelay (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on slider movement.
function sliderSource1_Callback(hObject, eventdata, handles)
% hObject    handle to sliderSource1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider


% --- Executes during object creation, after setting all properties.
function sliderSource1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to sliderSource1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


% --- Executes on slider movement.
function sliderSource2_Callback(hObject, eventdata, handles)
% hObject    handle to sliderSource2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider


% --- Executes during object creation, after setting all properties.
function sliderSource2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to sliderSource2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


% --- Executes on slider movement.
function sliderSource3_Callback(hObject, eventdata, handles)
% hObject    handle to sliderSource3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider


% --- Executes during object creation, after setting all properties.
function sliderSource3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to sliderSource3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


% --- Executes on button press in checkboxSource1.
function checkboxSource1_Callback(hObject, eventdata, handles)
% hObject    handle to checkboxSource1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkboxSource1


% --- Executes on button press in checkboxSource2.
function checkboxSource2_Callback(hObject, eventdata, handles)
% hObject    handle to checkboxSource2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkboxSource2


% --- Executes on button press in checkboxSource3.
function checkboxSource3_Callback(hObject, eventdata, handles)
% hObject    handle to checkboxSource3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkboxSource3



function editSimpleMSE_Callback(hObject, eventdata, handles)
% hObject    handle to editSimpleMSE (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of editSimpleMSE as text
%        str2double(get(hObject,'String')) returns contents of editSimpleMSE as a double


% --- Executes during object creation, after setting all properties.
function editSimpleMSE_CreateFcn(hObject, eventdata, handles)
% hObject    handle to editSimpleMSE (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function editEqsMSE_Callback(hObject, eventdata, handles)
% hObject    handle to editEqsMSE (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of editEqsMSE as text
%        str2double(get(hObject,'String')) returns contents of editEqsMSE as a double


% --- Executes during object creation, after setting all properties.
function editEqsMSE_CreateFcn(hObject, eventdata, handles)
% hObject    handle to editEqsMSE (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function editEsmMSE_Callback(hObject, eventdata, handles)
% hObject    handle to editEsmMSE (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of editEsmMSE as text
%        str2double(get(hObject,'String')) returns contents of editEsmMSE as a double


% --- Executes during object creation, after setting all properties.
function editEsmMSE_CreateFcn(hObject, eventdata, handles)
% hObject    handle to editEsmMSE (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function editSimpleVAR_Callback(hObject, eventdata, handles)
% hObject    handle to editSimpleVAR (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of editSimpleVAR as text
%        str2double(get(hObject,'String')) returns contents of editSimpleVAR as a double


% --- Executes during object creation, after setting all properties.
function editSimpleVAR_CreateFcn(hObject, eventdata, handles)
% hObject    handle to editSimpleVAR (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function editEqsVAR_Callback(hObject, eventdata, handles)
% hObject    handle to editEqsVAR (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of editEqsVAR as text
%        str2double(get(hObject,'String')) returns contents of editEqsVAR as a double


% --- Executes during object creation, after setting all properties.
function editEqsVAR_CreateFcn(hObject, eventdata, handles)
% hObject    handle to editEqsVAR (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function editEsmVAR_Callback(hObject, eventdata, handles)
% hObject    handle to editEsmVAR (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of editEsmVAR as text
%        str2double(get(hObject,'String')) returns contents of editEsmVAR as a double


% --- Executes during object creation, after setting all properties.
function editEsmVAR_CreateFcn(hObject, eventdata, handles)
% hObject    handle to editEsmVAR (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
