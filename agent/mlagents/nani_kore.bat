Taskkill /IM Neos.exe /F
start "" /d "C:\Program Files (x86)\Steam\steamapps\common\NeosVR\" run_neos_vrai_desktop.bat
timeout /t 50
mlagents-learn eco_npc_config.yaml --run-id=test_eco_demo --env="..\..\environment\neos\built_env\Unity Environment" --train --time-scale=2.0
Taskkill /IM Neos.exe /F

:loop

echo Ooops

Taskkill /IM Neos.exe /F
start "" /d "C:\Program Files (x86)\Steam\steamapps\common\NeosVR\" run_neos_vrai_desktop.bat
timeout /t 50
mlagents-learn eco_npc_config.yaml --env="..\..\environment\neos\built_env\Unity Environment" --run-id=test_eco_demo --train --load --time-scale=2.0
Taskkill /IM Neos.exe /F


goto loop