@echo off

:: Define the path to your virtual environment's activation script
set "VENV_ACTIVATE=C:\GitRepos\ProcessMining\venv3.13.9\Scripts\activate.bat"

:: Define the command to run your Streamlit application
set "STREAMLIT_CMD=streamlit run app.py"

echo Activating Python virtual environment...
call "%VENV_ACTIVATE%"

echo Running Streamlit application...
call %STREAMLIT_CMD%

:: Optionally, you can add 'pause' here to keep the console window open after the Streamlit app is closed
:: pause