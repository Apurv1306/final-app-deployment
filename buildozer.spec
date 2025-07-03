[app]

# (str) Title of your application
title = FaceApp Attendance

# (str) Package name. Use a reverse domain name unique to your app.
package.name = com.yourcompany.faceappattendance

# (str) Package domain (used with package.name to create unique identifier)
package.domain = yourcompany.com

# (list) Application requirements
# List all Python packages from your requirements.txt.
# python3 and hostpython3 are necessary for Python to run.
requirements = python3,hostpython3,flask,opencv-python,numpy,requests,flask-cors

# (str) Source code directory
# This assumes buildozer.spec is in the root of your Flask app's folder.
source.dir = .

# (list) Source files to include (leave empty to include all files in source.dir)
# Ensure all your project files, including HTML, CSS, JS, images, and Haar cascade, are included.
# 'xml' for haarcascade, 'mp3' for thank_you.mp3, 'html' for index.html
source.include_exts = py,png,jpg,kv,atlas,html,css,js,json,ttf,woff,woff2,xml,mp3

# (str) Application versioning (method 1)
version = 1.0.0

# (list) Permissions
# INTERNET is crucial for Flask to run and for the WebView to connect.
# CAMERA is vital for your face app.
# WRITE_EXTERNAL_STORAGE / READ_EXTERNAL_STORAGE are needed for saving known_faces.
android.permissions = INTERNET,CAMERA,WRITE_EXTERNAL_STORAGE,READ_EXTERNAL_STORAGE

# (int) Target Android API, should be as high as possible.
# As of July 2025, API 33 or 34 is recommended.
android.api = 34

# (int) Minimum Android API your app will support
android.minapi = 21

# (str) The Android arch to build for
android.arch = armeabi-v7a
# Good default for most devices. Can add 'arm64-v8a' for 64-bit devices: 'armeabi-v7a,arm64-v8a'

# (str) Bootstrap to use for android builds.
# THIS IS THE KEY PART FOR YOUR WEBVIEW APP
p4a.bootstrap = webview

# (int) port number to specify an explicit --port= p4a argument (eg for bootstrap flask)
# This port must match the one you use in python app.py (e.g., 5000)
p4a.port = 5000

# (str) The command to run when the app starts
# This will execute your Flask app's entry point
application.cmd = python "python app.py"

# (str) The URL that the webview will load
# This points to your local Flask server running inside the app
url = http://127.0.0.1:5000

# (bool) Automatically accept Android SDK licenses. Crucial for automated builds.
android.accept_sdk_license = True

# (dict) Environment variables to set when the Python app runs on Android
# These are crucial for your email functionality.
# You will set these as GitHub Secrets in your repository.
android.env_vars = \
    FACEAPP_EMAIL=__ENV_FACEAPP_EMAIL__,\
    FACEAPP_PASS=__ENV_FACEAPP_PASS__,\
    FACEAPP_ADMIN_EMAIL=__ENV_FACEAPP_ADMIN_EMAIL__

# Optional: App icon and presplash (loading screen)
# Create these files in your project root if you want custom ones.
#icon.filename = %(source.dir)s/icon.png
#presplash.filename = %(source.dir)s/presplash.png

[buildozer]
# (int) Log level (0 = error only, 1 = info, 2 = debug (with command output))
log_level = 2
