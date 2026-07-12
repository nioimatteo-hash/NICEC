@echo off
title NICEC 2026 – NIoi Corneal Entropy Calculator
color 0A

echo ============================================================
echo    NICEC 2026 – NIoi Corneal Entropy Calculator
echo    by Matteo Nioi
echo ============================================================
echo.

:: Controlla se Python è installato
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERRORE] Python non trovato.
    echo Assicurati che Python sia installato e nel PATH.
    echo.
    pause
    exit /b 1
)

:: Mostra la versione di Python in uso
python --version

:: Avvia NICEC2026.py
echo.
echo Avvio di NICEC2026...
python "%~dp0NICEC2026.py"

:: Se il programma termina (normale o con errore)
echo.
echo ============================================================
echo NICEC2026 terminato.
echo ============================================================
pause