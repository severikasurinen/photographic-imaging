# photographic-imaging
*THESIS WILL BE LINKED IN THE NEAR FUTURE*
Calibration scripts made as part of my B.Sc. thesis "Photographic imaging of solar cells". Currently only verified support for Windows, but should also support other platforms with minor adjustments.

# Installation
- Copy script files into a "Python" folder under a main imaging directory. Running main_script.py will automatically generate the rest of the required file structure.
- Download the ExifTool Windows Executable from https://exiftool.org/ and extract the exiftool.exe file into the Python folder.
- Create the folder "ICC Profiles" inside the Python folder and download at least the following profiles (and rename to match the name) into the folder.
  - sRGB.icc (https://www.color.org/srgbprofiles.xalter -> sRGB_v4_ICC_preference.icc)
  - Adobe.icc (https://www.adobe.com/support/downloads/iccprofiles/icc_eula_win_end.html -> Accept, Extract RGB -> AdobeRGB1998.icc)
  - ProPhoto.icc (https://sites.google.com/site/chromasoft/icmprofiles -> ICCProfiles.zip, extract ProPhoto.icc)
- Also copy the ICC Profiles into C:\Windows\System32\spool\drivers\color for use in other applications, such as Capture One and Adobe Photoshop.

Run the following commands in PowerShell or Command Prompt:
```powershell
pip install colour-science
```
```powershell
pip install matplotlib
```
```powershell
pip install opencv-python
```
```powershell
pip install natsort
```
```powershell
pip install PyExifTool
```

# Using the scripts
- Run the program by starting run.bat.
- Select the desired script mode and follow the instructions in photographic-imaging.pdf. The instructions have been designed specifically for the original system, but can be adjusted to fit another similar system by applying the findings of the linked thesis.
