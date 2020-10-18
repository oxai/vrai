tasklist /FI "IMAGENAME eq Neos.exe" 2>NUL | find /I /N "Neos.exe">NUL
if NOT "%ERRORLEVEL%"=="0" start "" /d "C:\Program Files (x86)\Steam\steamapps\common\NeosVR\" run_neos_vrai_desktop_rocks.bat 
tasklist /FI "IMAGENAME eq Neos.exe" 2>NUL | find /I /N "Neos.exe">NUL
if NOT "%ERRORLEVEL%"=="0" timeout /t 50
mlagents-learn eco_npc_config.yaml --run-id=test_eco_demo_new2 --env="..\..\environment\neos\built_env\Unity Environment" --resume --time-scale=2.0 && Taskkill /IM Neos.exe /F

:loop
Taskkill /IM Neos.exe /F
start "" /d "C:\Program Files (x86)\Steam\steamapps\common\NeosVR\" run_neos_vrai_desktop_rocks.bat 
timeout /t 50
mlagents-learn eco_npc_config.yaml --run-id=test_eco_demo_new2 --env="..\..\environment\neos\built_env\Unity Environment" --resume --time-scale=2.0 && Taskkill /IM Neos.exe /F
goto loop
