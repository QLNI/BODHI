# How to launch the BODHI dashboard

## Easiest — double-click

Double-click **`RUN.bat`** in this folder. It will:
1. Start the BODHI dashboard server
2. Open your browser to http://127.0.0.1:5000/

## Or from the command line

Open a terminal in this folder, then:

```bash
pip install -r requirements.txt
python chat_ui.py
```

Then open **http://127.0.0.1:5000/** in your browser.

---

## If the page looks wrong / images are broken

You might have another BODHI server already running from the `sk/` folder
on the same port.

- Windows: `taskkill /F /IM python.exe` (kills ALL Python processes)
- Or: find the other command-prompt window and press Ctrl-C there
- Then double-click `RUN.bat` again in this folder

---

## Stop the server

Close the terminal window, or press Ctrl-C in it.
