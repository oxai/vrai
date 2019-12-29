To run, you need first to install VRCModLoader via VRCModInstaller (found [here](https://github.com/Slaynash/VRChatModInstaller))

To build, open `vrcai.csproj` with Visual Studio, and click on the menu `Build->Build Solution`.

Then copy the generated dll `bin/Debug/vrcai.dll` and copy it inside the VRChat installation folder inside `Mods` folder (which should have some other dlls put there by VRCModInstaller).

Code lives in `vrcai.cs`. It has a few overrideable functions, like `OnApplicationStart`, `Update` (which runs at every Unity Update I think), and some others (see [here](https://github.com/Slaynash/VRCModLoader)).
Can also use methods from `VRCMenuUtilsAPI` to add buttons to menu (see commented out code).

Check the discord and github of vrctools for more info https://vrchat.survival-machines.fr/ and examples of mods for inspiration.

Use VRCTestingKit. Download release dll [here](https://github.com/FusGang/VRCTestingKit), and copy inside `Mods` folder inside VRChat installation folder.