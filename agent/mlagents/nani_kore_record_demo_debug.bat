Taskkill /IM Neos.exe /F
start "" /d "C:\Program Files (x86)\Steam\steamapps\common\NeosVR\" run_neos_vrai_desktop_record_demo.bat
timeout /t 50
mlagents-learn eco_npc_config.yaml --run-id=test_eco_demo --train --time-scale=2.0
Taskkill /IM Neos.exe /F