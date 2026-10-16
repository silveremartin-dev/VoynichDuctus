@echo off
title VoynichDuctus — Grand Glyph Explorer
echo =======================================================================
echo   VoynichDuctus : Grand Glyph Explorer (225 Folios + 371 Pages)
echo   Demarrage du serveur local sur http://localhost:8000/grand_glyph_explorer.html
echo =======================================================================
start "" http://localhost:8000/grand_glyph_explorer.html
py -3 -m http.server 8000 --directory "%~dp0output\atlas_dataset"
if errorlevel 1 (
    python -m http.server 8000 --directory "%~dp0output\atlas_dataset"
)
pause
